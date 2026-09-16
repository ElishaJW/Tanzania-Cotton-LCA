"""
plotting.py — Visualization functions for the Tanzania Cotton LCA.

Convention: this module is a *pure plotting layer*. All compute-heavy work
(extracting stage scores or district scores from LCA objects) is in
`compute_stage_scores()` / `compute_district_scores()` so that
01_scenarios can call them once with live LCA objects, save the results
to CSV, and 02_results can build all figures from the CSVs alone.

Public functions
----------------
draw_panel(ax, gdf, score_col, title, unit_label, ...)
    Choropleth panel helper.
draw_panel_shared(...)
    Same as draw_panel but uses a shared cmap/norm.
plot_cross_scenario_maps(...)
    Per-category figure with one panel per scenario, shared colour scale.
plot_spatial_map(panels, out_path, ...)
    Single-scenario 2x2 map. Takes pre-computed (gdf, score_col, ...) panels.
plot_contribution_breakdown(stage_df, district_df, cat_meta, ...)
    Stage breakdown (panel A) + top districts (panel B). Pre-computed inputs.
contribution_analysis(lca_obj, top_n)
    Compute top-N activity contributions (needs an LCA object).
plot_contribution_analysis(categories_results, out_path, ...)
    Multi-panel horizontal bar chart of top activities.
plot_dls_spider(coverage, out_path, ...)
    Doughnut/spider plot of DLS indicator coverage for one scenario.
plot_scenario_tradeoff(impacts_df, dls_df, ...)
    Wellbeing vs. biodiversity scatter across scenarios.
compute_dls_coverage_from_scenario(scenario)
    Helper to roll a Scenario into 6 DLS indicator coverage fractions.
compute_stage_scores(lca_obj)
    Per-stage LCA score breakdown (handles both bc.LCA and OneSpatialScaleLCA).
compute_district_scores(lca_obj, is_spatial, production_weights, tz_districts_gdf)
    Per-district score dict.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict

import bw2data as bd
import scipy.sparse


# ── Choropleth panel helpers ─────────────────────────────────────────────────

def draw_panel(ax, gdf_all, score_col, title, unit_label,
               cmap_name="YlOrRd", label_top_n=5):
    """Draw a choropleth map on `ax`."""
    impacted     = gdf_all[gdf_all[score_col].notna() & (gdf_all[score_col] > 0)].copy()
    not_impacted = gdf_all[~(gdf_all[score_col].notna() & (gdf_all[score_col] > 0))].copy()

    not_impacted.plot(ax=ax, color="#DCDCDC", edgecolor="#9A9A9A", linewidth=0.3, zorder=1)

    if len(impacted) == 0:
        ax.set_title(title + "\n(no non-zero data)", fontsize=9, fontweight="bold", pad=6)
        return

    vmin = impacted[score_col].min()
    vmax = impacted[score_col].max()
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    impacted.plot(ax=ax, column=score_col, cmap=cmap, norm=norm,
                  edgecolor="#444444", linewidth=0.4, zorder=2)
    gdf_all.dissolve().boundary.plot(ax=ax, color="#222222", linewidth=0.8, zorder=3)

    top_n = impacted.nlargest(label_top_n, score_col)
    for _, row in top_n.iterrows():
        cx = row.geometry.centroid.x
        cy = row.geometry.centroid.y
        lbl = row.get("ADM2_EN", "")
        ax.text(cx, cy, lbl, fontsize=4.5, ha="center", va="center",
                color="#111111", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
                zorder=5)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=22, shrink=0.7)
    cbar.set_label(unit_label, fontsize=7, labelpad=6)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))

    ax.set_title(title, fontsize=9, fontweight="bold", pad=6, linespacing=1.4)


def draw_panel_shared(ax, gdf_all, score_col, title, unit_label,
                      cmap, norm, label_top_n=5):
    """Choropleth using a pre-computed shared cmap/norm (no own colorbar)."""
    impacted     = gdf_all[gdf_all[score_col].notna() & (gdf_all[score_col] > 0)].copy()
    not_impacted = gdf_all[~(gdf_all[score_col].notna() & (gdf_all[score_col] > 0))].copy()

    not_impacted.plot(ax=ax, color="#DCDCDC", edgecolor="#9A9A9A", linewidth=0.3, zorder=1)

    if len(impacted) == 0:
        ax.set_title(title + "\n(no data)", fontsize=8, fontweight="bold", pad=4)
        return

    impacted.plot(ax=ax, column=score_col, cmap=cmap, norm=norm,
                  edgecolor="#444444", linewidth=0.3, zorder=2)
    gdf_all.dissolve().boundary.plot(ax=ax, color="#222222", linewidth=0.7, zorder=3)

    top_n = impacted.nlargest(label_top_n, score_col)
    for _, row in top_n.iterrows():
        cx = row.geometry.centroid.x
        cy = row.geometry.centroid.y
        lbl = row.get("ADM2_EN", "")
        ax.text(cx, cy, lbl, fontsize=4, ha="center", va="center",
                color="#111111", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
                zorder=5)

    ax.set_title(title, fontsize=8, fontweight="bold", pad=4, linespacing=1.3)


# ── Cross-scenario maps (one figure per impact category) ─────────────────────

def plot_cross_scenario_maps(
    category_name,
    unit_label,
    scenario_gdfs,
    tz_districts_gdf,
    out_path,
    cmap_name="YlOrRd",
    ncols=3,
    label_top_n=4,
):
    """One figure per impact category, all scenarios on a shared colour scale.

    scenario_gdfs : list of (display_name: str, gdf: GeoDataFrame, score_col: str).
    """
    import math

    n     = len(scenario_gdfs)
    nrows = math.ceil(n / ncols)
    cmap  = matplotlib.colormaps.get_cmap(cmap_name)

    all_vals = []
    for _, gdf, sc in scenario_gdfs:
        if sc in gdf.columns:
            vals = gdf[sc].dropna()
            vals = vals[vals > 0]
            all_vals.extend(vals.tolist())

    if not all_vals:
        print(f"No positive scores found for {category_name} — skipping map.")
        return

    vmin = 0.0
    vmax = max(all_vals)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5.5 * nrows + 0.8), dpi=150)
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flat

    for ax, (display_name, gdf, score_col) in zip(axes_flat, scenario_gdfs):
        gdf_utm = gdf.to_crs(epsg=32736) if gdf.crs is not None else gdf.set_crs(
            tz_districts_gdf.crs).to_crs(epsg=32736)
        draw_panel_shared(ax, gdf_utm, score_col,
                          title=display_name,
                          unit_label=unit_label,
                          cmap=cmap, norm=norm,
                          label_top_n=label_top_n)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal")

    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(unit_label, fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))

    fig.suptitle(
        f"Spatial Distribution of LCIA Impact — {category_name}\n"
        "Tanzania Cotton Supply Chain (all scenarios, shared colour scale)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 0.91, 1.0])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Cross-scenario map saved -> {out_path}")


# ── Combined-spatial-impact cross-scenario map ──────────────────────────────

def plot_combined_spatial_map(
    scenario_district_scores,
    cat_totals_by_scenario,
    tz_districts_gdf,
    out_path,
    cat_keys=("land_use", "water", "n_eutro", "p_eutro"),
    baseline_name="baseline",
    cmap_name="magma_r",
    ncols=3,
    label_top_n=5,
    score_col="combined_score",
    use_baseline_normaliser=True,
):
    """Cross-scenario map summing all regional impact categories per district.

    Each spatial impact category lives in different units (PDF·yr vs.
    PDF·m²·yr), so we normalise each category before summing. Two modes:

    • ``use_baseline_normaliser=True`` (default) — normalise every
      scenario's per-district scores by the BASELINE scenario's category
      total. Each category contributes 1.0 to the baseline national sum,
      so the baseline panel sums to ``len(cat_keys)``; other scenarios
      can rise (e.g. mfg expansion → water grows ~64×) or fall, and the
      colour scale is comparable across scenarios.

    • ``use_baseline_normaliser=False`` — normalise each scenario by its
      OWN category total. Each scenario panel sums to ``len(cat_keys)``;
      maps show only the WITHIN-scenario distribution of impact and are
      not directly comparable across scenarios.

    Parameters
    ----------
    scenario_district_scores : dict
        ``{scenario_name: {cat_key: {district_id: score}}}``
    cat_totals_by_scenario : dict
        ``{scenario_name: {cat_key: total_score}}``
    tz_districts_gdf : GeoDataFrame
        ADM2 districts with ``ADM2_PCODE`` and geometry.
    cat_keys : tuple of str
        Which category keys to include in the sum. Defaults to the four
        regionalised categories.
    baseline_name : str
        Scenario whose totals are used as cross-scenario normaliser (when
        ``use_baseline_normaliser`` is True).
    """
    import math

    scenarios = list(scenario_district_scores.keys())
    n     = len(scenarios)
    nrows = math.ceil(n / ncols)
    cmap  = matplotlib.colormaps.get_cmap(cmap_name)

    if use_baseline_normaliser:
        if baseline_name not in cat_totals_by_scenario:
            raise ValueError(
                f"Baseline scenario '{baseline_name}' not found in "
                f"cat_totals_by_scenario keys: {list(cat_totals_by_scenario)}")
        norm_totals = {
            ck: (abs(cat_totals_by_scenario[baseline_name].get(ck, 0.0)) or 1.0)
            for ck in cat_keys
        }
        norm_label  = f"normalised to {baseline_name} scenario totals"
    else:
        norm_totals = None
        norm_label  = "normalised to each scenario's own totals"

    # Compute combined score per (scenario, district)
    combined = {}   # {scenario: {district_id: combined_value}}
    for scen, per_cat in scenario_district_scores.items():
        if use_baseline_normaliser:
            totals = norm_totals
        else:
            totals = {
                ck: (abs(cat_totals_by_scenario.get(scen, {}).get(ck, 0.0)) or 1.0)
                for ck in cat_keys
            }
        all_dists = set()
        for ck in cat_keys:
            all_dists.update((per_cat.get(ck) or {}).keys())
        scen_combined = {}
        for d in all_dists:
            v = 0.0
            for ck in cat_keys:
                d_score = (per_cat.get(ck) or {}).get(d, 0.0)
                v += d_score / totals[ck]
            scen_combined[d] = v
        combined[scen] = scen_combined

    # Determine shared colour scale
    all_vals = []
    for scen, dvals in combined.items():
        all_vals.extend(v for v in dvals.values() if v > 0)
    if not all_vals:
        print(f"No positive combined-impact scores found — skipping map.")
        return
    vmin, vmax = 0.0, max(all_vals)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Build GeoDataFrames for each scenario
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5.5 * nrows + 0.8), dpi=150)
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flat

    for ax, scen in zip(axes_flat, scenarios):
        scen_combined = combined.get(scen, {})
        gdf = tz_districts_gdf.copy()
        gdf[score_col] = gdf["ADM2_PCODE"].map(scen_combined)
        gdf_utm = gdf.to_crs(epsg=32736) if gdf.crs is not None else gdf
        draw_panel_shared(
            ax, gdf_utm, score_col,
            title=scen.replace("_", " ").title(),
            unit_label="Combined regional impact (dimensionless, sum of normalised categories)",
            cmap=cmap, norm=norm,
            label_top_n=label_top_n,
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal")

    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(
        f"Sum of {len(cat_keys)} normalised regional impacts per district",
        fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))

    fig.suptitle(
        "Combined Regional Biodiversity Impact — Tanzania Cotton Supply Chain\n"
        f"Sum of {', '.join(cat_keys)} ({norm_label})",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 0.91, 1.0])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Combined regional-impact map saved -> {out_path}")


# ── Difference-vs-baseline cross-scenario map ───────────────────────────────

def plot_difference_spatial_map(
    scenario_district_scores,
    tz_districts_gdf,
    out_path,
    cat_keys=("land_use", "water", "n_eutro", "p_eutro"),
    baseline_name="baseline",
    baseline_cmap="YlOrRd",
    diff_cmap="RdBu_r",
    ncols=3,
    label_top_n=5,
):
    """Cross-scenario map: BASELINE panel = raw sum of regional impacts per
    district (PDF·yr).  Each non-baseline panel = scenario − baseline,
    summed across categories per district (PDF·yr).

    All four LC-IMPACT spatial endpoint categories produce per-district
    results in PDF·yr — the ``PDF·m²·year`` string in
    ``METHOD_LAND_USE_REGIONAL`` is a mis-labelled brightway unit string;
    the LC-IMPACT land-use CFs are PDF·m⁻², which combined with the
    ``m²·yr`` inventory amount give PDF·yr. The four categories are
    therefore summed directly, with no normalisation.

    Layout
    ------
    The baseline panel uses a sequential colourmap on its own scale.
    The other-scenario panels use a diverging colourmap (red = damage
    increase, blue = decrease) centred on zero, with a shared symmetric
    scale across all difference panels.

    Parameters
    ----------
    scenario_district_scores : dict
        ``{scenario_name: {cat_key: {district_id: score (PDF·yr)}}}``.
    cat_keys : tuple of str
        Categories to sum.  Default = the four regionalised PDF·yr
        endpoints (land_use, water, n_eutro, p_eutro).
    """
    import math

    if baseline_name not in scenario_district_scores:
        raise ValueError(
            f"Baseline scenario '{baseline_name}' not found in "
            f"scenario_district_scores: {list(scenario_district_scores)}")

    # Order: baseline first, then the rest in the order given by the dict
    other_scenarios = [s for s in scenario_district_scores if s != baseline_name]
    panel_order = [baseline_name] + other_scenarios

    # ── Step 1: per-district sum across categories, per scenario ─────────
    def _district_sum(scen):
        per_cat = scenario_district_scores[scen]
        all_dists = set()
        for ck in cat_keys:
            all_dists.update((per_cat.get(ck) or {}).keys())
        out = {}
        for d in all_dists:
            out[d] = sum((per_cat.get(ck) or {}).get(d, 0.0) for ck in cat_keys)
        return out

    sum_by_scen = {s: _district_sum(s) for s in panel_order}
    base_sum    = sum_by_scen[baseline_name]

    # ── Step 2: baseline raw sum and diff per non-baseline scenario ──────
    diff_by_scen = {}
    for s in other_scenarios:
        ssum = sum_by_scen[s]
        all_dists = set(ssum) | set(base_sum)
        diff_by_scen[s] = {
            d: ssum.get(d, 0.0) - base_sum.get(d, 0.0)
            for d in all_dists
        }

    # ── Step 3: colour scales ────────────────────────────────────────────
    base_vals = [v for v in base_sum.values() if v and v > 0]
    if not base_vals:
        print("No positive baseline values found — skipping difference map.")
        return
    base_norm = Normalize(vmin=0.0, vmax=max(base_vals))
    base_cmap = matplotlib.colormaps.get_cmap(baseline_cmap)

    diff_vals = []
    for s in other_scenarios:
        diff_vals.extend(v for v in diff_by_scen[s].values() if v)
    if diff_vals:
        diff_max = max(abs(v) for v in diff_vals)
    else:
        diff_max = 1.0
    diff_norm = Normalize(vmin=-diff_max, vmax=diff_max)
    d_cmap    = matplotlib.colormaps.get_cmap(diff_cmap)

    # ── Step 4: lay out figure ───────────────────────────────────────────
    n     = len(panel_order)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5.5 * nrows + 0.8), dpi=150)
    fig.patch.set_facecolor("white")
    axes_flat = list(np.array(axes).flat)

    for ax, scen in zip(axes_flat, panel_order):
        gdf = tz_districts_gdf.copy()
        if scen == baseline_name:
            gdf["score"] = gdf["ADM2_PCODE"].map(base_sum)
            gdf_utm = gdf.to_crs(epsg=32736) if gdf.crs is not None else gdf
            draw_panel_shared(
                ax, gdf_utm, "score",
                title=f"{scen.replace('_', ' ').title()} — total regional impact",
                unit_label="Sum of regional impacts (PDF·yr)",
                cmap=base_cmap, norm=base_norm,
                label_top_n=label_top_n,
            )
        else:
            gdf["score"] = gdf["ADM2_PCODE"].map(diff_by_scen[scen])
            gdf_utm = gdf.to_crs(epsg=32736) if gdf.crs is not None else gdf
            draw_panel_shared(
                ax, gdf_utm, "score",
                title=f"{scen.replace('_', ' ').title()} − Baseline (Δ)",
                unit_label="Δ Sum of regional impacts (PDF·yr)",
                cmap=d_cmap, norm=diff_norm,
                label_top_n=label_top_n,
            )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # ── Step 5: two colourbars (baseline + diff) ─────────────────────────
    sm_b = ScalarMappable(cmap=base_cmap, norm=base_norm); sm_b.set_array([])
    sm_d = ScalarMappable(cmap=d_cmap,    norm=diff_norm); sm_d.set_array([])

    cbar_b_ax = fig.add_axes([0.92, 0.55, 0.015, 0.30])
    cbar_b    = fig.colorbar(sm_b, cax=cbar_b_ax)
    cbar_b.set_label("Baseline total\n(PDF·yr)", fontsize=8.5, labelpad=6)
    cbar_b.ax.tick_params(labelsize=7.5)
    cbar_b.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    cbar_b.ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))

    cbar_d_ax = fig.add_axes([0.92, 0.15, 0.015, 0.30])
    cbar_d    = fig.colorbar(sm_d, cax=cbar_d_ax)
    cbar_d.set_label("Δ vs baseline\n(red = ↑ damage)", fontsize=8.5, labelpad=6)
    cbar_d.ax.tick_params(labelsize=7.5)
    cbar_d.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    cbar_d.ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))

    fig.suptitle(
        "Regional Biodiversity Impact — Baseline + Per-Scenario Δ\n"
        f"Sum of {', '.join(cat_keys)} (PDF·yr); diff panels share a symmetric scale",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 0.91, 1.0])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Difference map saved -> {out_path}")


# ── 2x2 spatial map (single scenario) ────────────────────────────────────────

def plot_spatial_map(
    panels,
    out_path,
    suptitle="Spatial Distribution of LCIA Impact Indicators — Tanzania Cotton Supply Chain",
    footnote=(
        "Climate change (LC-IMPACT, RCP2.6) and freshwater ecotoxicity (USEtox 2.1) "
        "use global CFs — no district-level spatial variation, not mapped. "
        "N and P eutrophication use diffuse characterization factors (CF_marginal_diffuse): "
        "CFs incorporate aquatic fate factor, so emission amount equals field application rate."
    ),
):
    """Generate a 2x2 district-level LCIA spatial map from pre-computed
    GeoDataFrames.

    Parameters
    ----------
    panels : list of dict (length 4). Each dict must contain:
        'gdf'        : GeoDataFrame already merged with district geometry
                       (one row per district, NaN where no data)
        'score_col'  : numeric column name to plot
        'title'      : panel title
        'unit_label' : colorbar label
        'cmap_name'  : matplotlib colormap (optional, default 'YlOrRd')
        'label_top_n': number of top districts to label (optional, default 5)
    out_path : str
    """
    if len(panels) != 4:
        raise ValueError(f"plot_spatial_map expects 4 panels, got {len(panels)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 16), dpi=150)
    fig.patch.set_facecolor("white")

    for ax, p in zip(axes.flat, panels):
        gdf = p["gdf"]
        try:
            gdf = gdf.to_crs(epsg=32736)
        except Exception:
            pass
        draw_panel(
            ax, gdf, p["score_col"],
            title=p["title"],
            unit_label=p["unit_label"],
            cmap_name=p.get("cmap_name", "YlOrRd"),
            label_top_n=p.get("label_top_n", 5),
        )

    for ax in axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_aspect("equal")

    if footnote:
        fig.text(
            0.5, 0.01, footnote,
            ha="center", va="bottom", fontsize=8, style="italic",
            color="#555555", wrap=True,
        )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Spatial map saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE HELPERS — only called from 01_scenarios with live LCA objects.
# Outputs are dicts/DataFrames that can be persisted to CSV and consumed by
# 02_results without ever touching brightway again.
# ─────────────────────────────────────────────────────────────────────────────

# Supply chain stage code prefixes -> display labels
# Foreground activity code -> stage label, matched by str.startswith().
#
# Two code schemes exist in src/inventory.py:
#   * district-level activities use f"{prefix}_{district_id}" (cotton_prod_*,
#     lint_gin_*, seed_gin_*, textile_*)
#   * national stub processes use short codes assigned in _proc():
#     act1 cotton_production, act2a/act2b ginning, act3 textile_production,
#     act6a/act6b oil milling, act7 export_market, act8 domestic_market
#
# Oil milling and the markets exist ONLY as national stubs, so they must be
# matched on their "act6"/"act7"/"act8" codes. Matching them on the process
# NAME (as the previous "oil_milling"/"export_market"/"domestic_market" keys
# did) never fired, which mattered: in the spatial branch below an unmatched
# activity is dropped and the remaining stages are then rescaled to the true
# total, so oil milling's direct water flows were being silently redistributed
# onto the other stages.
#
# The national aggregator stubs (act1, act2a, act2b, act3) are deliberately
# absent: they carry only technosphere edges, so they contribute no direct
# burden and would appear as empty rows.
STAGE_CODES = {
    "cotton_prod": "Cotton Farming",
    "lint_gin":    "Lint Ginning",
    "seed_gin":    "Seed Ginning",
    "textile_":    "Textile Mfg.",
    "act6":        "Oil Milling",      # act6a (oil) + act6b (cake)
    "act7":        "Export Market",
    "act8":        "Domestic Market",
}


def compute_stage_scores(lca_obj):
    """Extract {stage_label: total_score} from an LCA object.

    For standard bc.LCA objects the characterized_inventory matrix is used
    directly. For bwr.OneSpatialScaleLCA objects that matrix is all-zero
    (location-qualified CFs are handled outside the standard matrix), so we
    fall back to computing biosphere_matrix x supply_array x average_CF, then
    rescale to the actual spatial score so relative stage shares are correct.
    """
    import bw2regional as _bwr

    is_spatial = isinstance(lca_obj, _bwr.OneSpatialScaleLCA)

    if not is_spatial:
        ci = lca_obj.characterized_inventory
        _rows, cols_idx, vals = scipy.sparse.find(ci)
        rev_act = {v: k for k, v in lca_obj.activity_dict.items()}
        scores = defaultdict(float)
        bg_total = 0.0
        for col_idx, val in zip(cols_idx, vals):
            act_id = rev_act.get(col_idx)
            if act_id is None:
                continue
            try:
                act_node = bd.get_node(id=act_id)
                code = act_node.get("code", "")
            except Exception:
                continue
            matched = False
            for prefix, label in STAGE_CODES.items():
                if code.startswith(prefix):
                    scores[label] += val
                    matched = True
                    break
            if not matched:
                # Background ecoinvent activity — direct emissions weighted by CF.
                # For climate/ecotox most of the score lives here; for spatial
                # methods (land use, water, N, P) bg activities have CF=0 so this
                # stays ~0 and the bar doesn't appear.
                bg_total += val
        if bg_total != 0.0:
            scores["Supply Chain (bg)"] = bg_total
        return dict(scores)

    # Spatial LCA fallback: approximate stage shares using average CFs per flow,
    # then rescale to the actual spatial score so relative contributions are correct.
    try:
        method_cfs = bd.Method(lca_obj.method).load()
        cf_sum = defaultdict(float)
        cf_cnt = defaultdict(int)
        for entry in method_cfs:
            # bw2data 4.x stores flow keys as integer node IDs, not (db, code) tuples.
            # Regionalized methods have 3-tuples (flow_id, cf, location); non-spatial
            # have 2-tuples (flow_id, cf).  Either way, entry[0] is the flow node ID.
            fid = entry[0] if isinstance(entry[0], int) else entry[0][1]
            cf  = entry[1]
            if cf and cf == cf:  # skip NaN/None
                cf_sum[fid] += cf
                cf_cnt[fid] += 1
        avg_cf = {fid: cf_sum[fid] / cf_cnt[fid] for fid in cf_sum}

        try:
            fg_codes = {a.id: a.get("code", "")
                        for a in bd.Database("foreground")}
        except Exception:
            fg_codes = {}

        # biosphere_dict / activity_dict are backward-compat properties on LCA
        # (bw2calc 2.x).  Both map node_id → matrix_index.
        rev_bio = {v: k for k, v in lca_obj.biosphere_dict.items()}
        rev_act = {v: k for k, v in lca_obj.activity_dict.items()}
        B   = lca_obj.biosphere_matrix.tocsc()
        s   = lca_obj.supply_array
        scores = defaultdict(float)

        for col_idx in range(B.shape[1]):
            act_id = rev_act.get(col_idx)
            if act_id is None:
                continue
            code = fg_codes.get(act_id)
            if code is None:
                try:
                    code = bd.get_node(id=act_id).get("code", "")
                except Exception:
                    continue

            stage_label = None
            for prefix, label in STAGE_CODES.items():
                if code.startswith(prefix):
                    stage_label = label
                    break
            if stage_label is None:
                continue

            start, end = B.indptr[col_idx], B.indptr[col_idx + 1]
            for ptr in range(start, end):
                bio_id = rev_bio.get(B.indices[ptr])
                cf = avg_cf.get(bio_id, 0.0)
                scores[stage_label] += cf * float(B.data[ptr]) * float(s[col_idx])

        approx = sum(scores.values())
        if approx != 0 and lca_obj.score != 0:
            scale = lca_obj.score / approx
            scores = {k: v * scale for k, v in scores.items()}

        return dict(scores)
    except Exception as _exc:
        import traceback
        print(f"  [compute_stage_scores] spatial fallback failed: {_exc}")
        traceback.print_exc()
        return {}


def compute_district_scores(lca_obj, is_spatial, production_weights, tz_districts_gdf):
    """Extract {district_id: total_score} from an LCA object.

    For spatial LCAs, bwr.geodataframe_inv_spatial_scale() returns a GeoDataFrame
    with columns ``score_abs`` (total characterised score per location) and
    ``location_key`` (stringified tuple e.g. ``"('tz_districts', 'TZ001')"``).
    There are no original shapefile columns (ADM2_PCODE etc.) in that frame, so
    the old approach of looking for a shared column name with tz_districts_gdf
    always returned ``None``.  We now parse the district ID from ``location_key``
    with ``ast.literal_eval``.
    """
    import ast

    if not is_spatial:
        return {d: w * lca_obj.score for d, w in production_weights.items()}

    try:
        # sum_flows=True (default) collapses biosphere rows → one row per spatial unit
        gdf = lca_obj.geodataframe_inv_spatial_scale()
        if gdf is None or len(gdf) == 0:
            return {}

        result = {}
        for _, row in gdf.iterrows():
            score = row.get("score_abs", None)
            if score is None or (score != score):  # None or NaN
                continue
            score = float(score)
            if score == 0.0:
                continue

            loc_str = str(row.get("location_key", ""))
            try:
                # location_key is stored as str(tuple), e.g. "('tz_districts', 'TZ001')"
                loc = ast.literal_eval(loc_str)
                district_id = loc[1] if isinstance(loc, tuple) else loc_str
            except Exception:
                district_id = loc_str

            result[district_id] = result.get(district_id, 0.0) + score

        return result
    except Exception as _exc:
        import traceback
        print(f"  [compute_district_scores] spatial approach failed: {_exc}")
        traceback.print_exc()
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# CONTRIBUTION BREAKDOWN (panel A: stages, panel B: top districts)
# ─────────────────────────────────────────────────────────────────────────────

def plot_contribution_breakdown(
    stage_scores_per_cat,
    district_scores_per_cat,
    cat_totals,
    cat_meta,
    tz_districts_gdf,
    out_path,
    top_district_n=15,
    baseline_cat_totals=None,
):
    """Stage breakdown (panel A) + top-N districts (panel B), pre-computed inputs.

    Parameters
    ----------
    stage_scores_per_cat : dict {cat_label: {stage_label: score}}
    district_scores_per_cat : dict {cat_label: {district_id: score}}
    cat_totals : dict {cat_label: total_score}
        Per-scenario category totals.
    cat_meta : list of (cat_label, cmap_name)
        Defines display order and colours for the panels.
    tz_districts_gdf : GeoDataFrame
        For mapping district ids -> human-readable names.
    out_path : str
    top_district_n : int
    baseline_cat_totals : dict {cat_label: total_score} or None, optional
        If provided, panel B normalises each category by the BASELINE
        scenario's total (not the current scenario's total). This makes
        bar heights comparable across scenarios — e.g. manufacturing
        expansion shows much taller water-impact bars in the textile
        districts because the absolute water score grew, even though the
        within-scenario fractional shares are unchanged. If omitted (the
        original behaviour), each category is normalised by its own
        scenario total and every scenario's bars sum to ~n_categories.
    """
    cat_labels = [lbl for lbl, _ in cat_meta]

    # --- Panel A active stages ----------------------------------------------
    # Foreground stages first (in canonical order), then any extra stages we
    # picked up from the data (e.g. "Supply Chain (bg)" for climate/ecotox).
    fg_stages = list(STAGE_CODES.values())
    extra_stages = []
    for lbl in cat_labels:
        for sg in stage_scores_per_cat.get(lbl, {}):
            if sg not in fg_stages and sg not in extra_stages:
                extra_stages.append(sg)
    stages = fg_stages + extra_stages
    active_stages = [
        sg for sg in stages
        if any(stage_scores_per_cat.get(lbl, {}).get(sg, 0) != 0
               for lbl in cat_labels)
    ]
    if not active_stages:
        active_stages = stages

    # --- Panel B top districts ----------------------------------------------
    all_dist_ids = set()
    for d in district_scores_per_cat.values():
        all_dist_ids.update(d.keys())

    # Choose normaliser. If baseline totals are provided, use them so bar
    # heights are CROSS-SCENARIO comparable (e.g. mfg expansion shows
    # much taller water-related bars than baseline). Otherwise fall back
    # to per-scenario totals and panel B sums to ~1.0 per category.
    norm_source = baseline_cat_totals if baseline_cat_totals is not None else cat_totals
    norm_cat_totals = {
        lbl: (abs(norm_source.get(lbl, 0.0)) or 1.0) for lbl in cat_labels
    }

    district_norm_total = {
        dist_id: sum(
            district_scores_per_cat.get(lbl, {}).get(dist_id, 0.0)
            / norm_cat_totals[lbl]
            for lbl in cat_labels
        )
        for dist_id in all_dist_ids
    }
    top_ids = sorted(district_norm_total, key=district_norm_total.get,
                     reverse=True)[:top_district_n]
    id_to_name = {row.get("ADM2_PCODE", ""): row.get("ADM2_EN", row.get("ADM2_PCODE", ""))
                  for _, row in tz_districts_gdf.iterrows()}
    dist_labels_b = [id_to_name.get(d, d) for d in top_ids]

    # --- Draw figure ---------------------------------------------------------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    fig.patch.set_facecolor("white")

    n_stages = len(active_stages)
    n_cats   = len(cat_labels)
    x        = np.arange(n_cats)
    width    = 0.7 / n_stages
    stage_colors = plt.cm.tab10.colors

    for i, stage in enumerate(active_stages):
        vals_a  = [stage_scores_per_cat.get(lbl, {}).get(stage, 0.0)
                   for lbl in cat_labels]
        offsets = x + (i - n_stages / 2 + 0.5) * width
        ax_a.bar(offsets, vals_a, width=width * 0.9,
                 label=stage, color=stage_colors[i % len(stage_colors)],
                 edgecolor="white", linewidth=0.4)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cat_labels, fontsize=9, rotation=15, ha="right")
    ax_a.set_title("A.  LCIA Score by Supply Chain Stage and Impact Category",
                   fontsize=11, fontweight="bold")
    ax_a.set_ylabel("LCIA score (native units per category)", fontsize=9)
    ax_a.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax_a.yaxis.get_major_formatter().set_powerlimits((-2, 2))
    ax_a.legend(fontsize=7.5, loc="upper right", title="Stage", title_fontsize=8)
    ax_a.grid(axis="y", linestyle="--", linewidth=0.3, color="#CCCCCC")
    ax_a.set_xlabel("Impact category", fontsize=9, labelpad=6)

    lefts_b = np.zeros(len(top_ids))
    cmap_b  = plt.cm.Set2.colors

    for i, lbl in enumerate(cat_labels):
        vals_b = [district_scores_per_cat.get(lbl, {}).get(d, 0.0)
                  / norm_cat_totals[lbl] for d in top_ids]
        ax_b.barh(dist_labels_b, vals_b, left=lefts_b,
                  label=lbl, color=cmap_b[i % len(cmap_b)],
                  edgecolor="white", linewidth=0.4)
        lefts_b += np.array(vals_b)

    ax_b.set_title(f"B.  Top-{top_district_n} Districts: Combined Normalised Contribution",
                   fontsize=11, fontweight="bold")
    _xlabel_b = (
        "Sum of fractional contributions (each category / BASELINE total — "
        "comparable across scenarios)"
        if baseline_cat_totals is not None else
        "Sum of fractional contributions (each category normalised to its own scenario total)"
    )
    ax_b.set_xlabel(_xlabel_b, fontsize=8.5, labelpad=6)
    ax_b.invert_yaxis()
    ax_b.tick_params(axis="y", labelsize=8)
    ax_b.legend(fontsize=7.5, loc="lower right", title="Category", title_fontsize=8)
    ax_b.grid(axis="x", linestyle="--", linewidth=0.3, color="#CCCCCC")

    fig.suptitle(
        "Supply Chain Contribution Breakdown — Tanzania Cotton LCA (All Impact Categories)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Contribution figure saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FULL ACTIVITY CONTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def contribution_analysis(lca_obj, top_n=20):
    """Return a DataFrame of the top-N contributing activities (uses LCA object)."""
    import bw2analyzer as ba

    ca = ba.ContributionAnalysis()
    raw = ca.annotated_top_processes(lca_obj, limit=top_n)
    total = abs(lca_obj.score) if lca_obj.score else 1.0

    rows = []
    for rank, (score, _supply, act) in enumerate(raw, 1):
        rows.append({
            "rank":              rank,
            "activity":          act.get("name", ""),
            "reference_product": act.get("reference product", ""),
            "database":          act.get("database", ""),
            "location":          str(act.get("location", "")),
            "score":             float(score),
            "pct":               100 * abs(float(score)) / total,
        })
    return pd.DataFrame(rows)


def plot_contribution_analysis(categories_results, out_path, top_n=15):
    """Multi-panel horizontal bar chart of top activities per impact category.

    Parameters
    ----------
    categories_results : list of (label, df)
        Each tuple is (impact category label, DataFrame from contribution_analysis()).
    out_path : str
    top_n : int
    """
    n = len(categories_results)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(18, nrows * 5 + 1), dpi=120)
    fig.patch.set_facecolor("white")
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    DB_COLORS = {
        "foreground":            "#E07B39",
        "ecoinvent-3.12-cutoff": "#4C72B0",
    }
    DEFAULT_COLOR = "#888888"

    for ax, (label, df) in zip(axes_flat, categories_results):
        plot_df = df.head(top_n).copy()
        plot_df["short_name"] = plot_df.apply(
            lambda r: (r["activity"][:45] + "…" if len(r["activity"]) > 45
                       else r["activity"]) +
                      (f"  [{r['location']}]" if r["location"] not in ("", "None") else ""),
            axis=1,
        )
        colors = [DB_COLORS.get(db, DEFAULT_COLOR) for db in plot_df["database"]]

        bars = ax.barh(plot_df["short_name"], plot_df["pct"],
                       color=colors, edgecolor="white", linewidth=0.4)
        ax.invert_yaxis()
        ax.set_xlabel("Contribution (%)", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="x", linestyle="--", linewidth=0.3, color="#CCCCCC")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

        for bar, (_, row) in zip(bars, plot_df.iterrows()):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{row['score']:.2e}", va="center", fontsize=6.5, color="#333333")

    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)

    legend_patches = [
        mpatches.Patch(color=DB_COLORS["foreground"],            label="Foreground (this study)"),
        mpatches.Patch(color=DB_COLORS["ecoinvent-3.12-cutoff"], label="Background (ecoinvent 3.12)"),
        mpatches.Patch(color=DEFAULT_COLOR,                       label="Other"),
    ]
    fig.legend(handles=legend_patches, loc="lower right",
               fontsize=8, title="Database", title_fontsize=9,
               bbox_to_anchor=(0.98, 0.01))

    fig.suptitle(
        "Full Activity Contribution Analysis — Tanzania Cotton LCA",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Contribution analysis figure saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# DLS SPIDER PLOT
# ─────────────────────────────────────────────────────────────────────────────

def compute_dls_coverage_from_scenario(scenario) -> dict:
    """Return {indicator_label: coverage_fraction} for a given scenario."""
    a = scenario.run_dls()
    ctx = scenario.national
    A = ctx.national_area_ha

    kcal_nat    = a.nutrition["kcal_total_ha"] * A
    protein_nat = a.nutrition["protein_kg_ha"] * A
    frac_kcal    = kcal_nat / (ctx.population * scenario.nutrition.kcal_per_cap_yr)
    frac_protein = protein_nat / (ctx.population
                                  * scenario.nutrition.protein_per_cap_yr_kg)

    fabric_nat  = a.clothing["fabric_kg_ha"] * A
    threshold   = (scenario.clothing.kg_fabric_per_cap_yr
                   * scenario.cotton_share_of_clothing * ctx.population)
    frac_clothing = fabric_nat / threshold

    frac_farm_li = a.farm_income["living_income_coverage"]

    fte_nat = a.employment["total_fte_ha"] * A
    frac_jobs = fte_nat / ctx.rural_labor_force

    return {
        "Nutrition\n(kcal)":            frac_kcal,
        "Nutrition\n(protein)":         frac_protein,
        "Clothing":                     frac_clothing,
        "Farm income\n(living-income)": frac_farm_li,
        "Employment\n(rural labour)":   frac_jobs,
    }


def compute_dls_coverage(yield_kg_ha: float = 400.0,
                         cotton_share_of_clothing: float | None = None) -> dict:
    """Default-parameter DLS coverage helper (used outside scenarios)."""
    from src.needs import (
        assess_needs, NationalContext, ClothingParams, NutritionParams,
    )

    kwargs = {"yield_kg_ha": yield_kg_ha}
    if cotton_share_of_clothing is not None:
        kwargs["cotton_share_of_clothing"] = cotton_share_of_clothing
    a   = assess_needs(**kwargs)
    ctx = NationalContext()
    A   = ctx.national_area_ha

    share = (cotton_share_of_clothing
             if cotton_share_of_clothing is not None
             else 0.5)

    kcal_nat    = a.nutrition["kcal_total_ha"] * A
    protein_nat = a.nutrition["protein_kg_ha"] * A
    frac_kcal    = kcal_nat / (ctx.population * NutritionParams().kcal_per_cap_yr)
    frac_protein = protein_nat / (ctx.population
                                  * NutritionParams().protein_per_cap_yr_kg)

    fabric_nat  = a.clothing["fabric_kg_ha"] * A
    threshold   = (ClothingParams().kg_fabric_per_cap_yr
                   * share * ctx.population)
    frac_clothing = fabric_nat / threshold

    frac_farm_li = a.farm_income["living_income_coverage"]

    fte_nat = a.employment["total_fte_ha"] * A
    frac_jobs = fte_nat / ctx.rural_labor_force

    return {
        "Nutrition\n(kcal)":            frac_kcal,
        "Nutrition\n(protein)":         frac_protein,
        "Clothing":                     frac_clothing,
        "Farm income\n(living-income)": frac_farm_li,
        "Employment\n(rural labour)":   frac_jobs,
    }


def plot_dls_spider(
    coverage: dict,
    out_path,
    title: str = "DLS indicator coverage — Tanzania cotton industry (baseline)",
):
    """Doughnut/spider plot: each axis = fraction of national DLS floor."""
    labels = list(coverage.keys())
    values = np.array(list(coverage.values()), dtype=float)
    n      = len(labels)

    theta         = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_closed  = np.concatenate([theta, theta[:1]])
    values_closed = np.concatenate([values, values[:1]])
    r_max         = max(1.2, values.max() * 1.08)

    fig = plt.figure(figsize=(10, 9))
    ax  = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rgrids([0.25, 0.50, 0.75, 1.00],
                  labels=["25%", "50%", "75%", "100%"],
                  angle=90, fontsize=11, color="#666666")
    ax.set_ylim(0, r_max)

    ring_theta = np.linspace(0, 2 * np.pi, 361)
    ax.plot(ring_theta, np.ones_like(ring_theta),
            linestyle="--", color="#1F4E79", linewidth=1.5,
            zorder=2, label="DLS floor (100%)")

    shortfall = np.minimum(values_closed, 1.0)
    ax.fill_between(theta_closed, shortfall, 1.0,
                    color="#1F4E79", alpha=0.08, zorder=1,
                    label="Shortfall to DLS floor")

    ax.plot(theta_closed, values_closed, color="#C65911",
            linewidth=2, zorder=3)
    ax.fill(theta_closed, values_closed, color="#C65911",
            alpha=0.35, zorder=3, label="Cotton industry coverage")

    if np.any(values > 1.0):
        overshoot = np.where(values_closed > 1.0, values_closed, np.nan)
        ax.fill_between(theta_closed, 1.0, overshoot,
                        color="#C65911", alpha=0.18, zorder=2)

    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=13)

    for t, v in zip(theta, values):
        ax.text(t, min(v, r_max) + 0.05 * r_max,
                f"{v*100:.1f}%",
                ha="center", va="center", fontsize=12,
                color="#8C3B00", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="none", alpha=0.8),
                zorder=4)

    ax.set_title(title, fontsize=16, pad=24)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05),
              fontsize=11, frameon=False)
    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"DLS spider plot saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO TRADE-OFF SCATTER
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_pct_change(df, value_col, group_col, scenario_col, baseline_name,
                          mode: str = "mean"):
    """Return {scenario: aggregated % change vs baseline across `group_col`}.

    Two aggregation modes:
      • ``"mean"`` — compute % change PER category, then take the mean. Each
        category counts equally regardless of magnitude. Tends to over-weight
        scenarios that move many categories by small amounts.
      • ``"sum"``  — sum the raw scores across categories first, then take
        % change of the sum. Categories with larger absolute magnitude
        (e.g. ecotox) dominate the result, which more faithfully reflects
        the change in total biodiversity damage.
    """
    pivot = df.pivot_table(index=group_col, columns=scenario_col,
                           values=value_col, aggfunc="first")
    if baseline_name not in pivot.columns:
        raise ValueError(
            f"Baseline scenario '{baseline_name}' not found. "
            f"Available: {list(pivot.columns)}"
        )
    if mode == "sum":
        totals  = pivot.sum(axis=0)        # one number per scenario
        base    = totals.get(baseline_name, np.nan)
        if not base or base != base or base == 0:
            return {s: float("nan") for s in totals.index}
        return ((totals - base) / base * 100.0).to_dict()
    elif mode == "mean":
        base = pivot[baseline_name].replace(0, np.nan)
        pct  = pivot.sub(base, axis=0).div(base, axis=0) * 100.0
        return pct.mean(axis=0).to_dict()
    elif mode == "geomean":
        # Geometric mean of per-category fold-changes, returned as % change.
        # Averaging ratios arithmetically is biased toward the largest ratio;
        # the geometric mean (arithmetic mean in log space) is the correct
        # average for fold-changes and prevents any single category from
        # dominating. Non-positive values cannot be log-transformed and are
        # dropped from the average.
        base   = pivot[baseline_name].replace(0, np.nan)
        ratios = pivot.div(base, axis=0).where(lambda r: r > 0)
        geo    = np.exp(np.log(ratios).mean(axis=0, skipna=True))
        return ((geo - 1.0) * 100.0).to_dict()
    else:
        raise ValueError(f"Unknown aggregation mode: {mode!r}")


def plot_scenario_tradeoff(
    impacts_df,
    dls_df,
    out_path,
    baseline_name: str = "baseline",
    scenario_display: dict | None = None,
    scenario_colors: dict | None = None,
    exclude_categories=("Ratio", "site-generic"),
    impacts_mode: str = "geomean",   # "geomean" (recommended), "sum", or "mean"
    wellbeing_mode: str = "mean",    # DLS coverages are bounded fractions, "mean" stays interpretable
    title: str = ("Scenario trade-off — aggregated change vs. baseline\n"
                  "Tanzania cotton supply chain"),
):
    """Scatter plot: aggregated % change in wellbeing vs. biodiversity impact.

    See :func:`_aggregate_pct_change` for the meaning of ``impacts_mode``
    and ``wellbeing_mode``.

    ``impacts_mode="geomean"`` is the recommended default. The two simpler
    alternatives each fail on this dataset:

      • ``"sum"`` is dominated by freshwater ecotoxicity, which is ~99.8% of
        the baseline total in absolute PDF*yr, so the aggregate collapses to
        a single category.
      • ``"mean"`` is dominated by water consumption, whose baseline is
        effectively zero under rainfed production; any irrigation scenario
        then produces a ~200x fold-change that swamps the average.

    The geometric mean of per-category fold-changes avoids both failure
    modes. Note that it weights every impact category equally regardless of
    absolute magnitude, so it is a MULTI-CRITERIA index of change, not a
    measure of total biodiversity damage — label axes accordingly.
    """
    impacts = impacts_df.copy()
    if exclude_categories:
        mask = ~impacts["category"].astype(str).str.contains(
            "|".join(exclude_categories), case=False, regex=True)
        impacts = impacts[mask]

    bio_pct = _aggregate_pct_change(
        impacts, value_col="score",    group_col="category",
        scenario_col="scenario",       baseline_name=baseline_name,
        mode=impacts_mode)
    wb_pct  = _aggregate_pct_change(
        dls_df,  value_col="coverage", group_col="indicator",
        scenario_col="scenario",       baseline_name=baseline_name,
        mode=wellbeing_mode)

    scenarios = [s for s in dict.fromkeys(
        list(impacts["scenario"]) + list(dls_df["scenario"]))
        if s in bio_pct and s in wb_pct]

    if scenario_display is None:
        scenario_display = {s: s.replace("_", " ").title() for s in scenarios}
    if scenario_colors is None:
        palette = plt.cm.Set1.colors
        scenario_colors = {s: palette[i % len(palette)] for i, s in enumerate(scenarios)}

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    fig.patch.set_facecolor("white")

    xs = [wb_pct[s]  for s in scenarios]
    ys = [bio_pct[s] for s in scenarios]

    x_pad = max(2.0, 0.15 * max(abs(min(xs)), abs(max(xs)), 1.0))
    y_pad = max(2.0, 0.15 * max(abs(min(ys)), abs(max(ys)), 1.0))
    x_lo, x_hi = min(xs) - x_pad, max(xs) + x_pad
    y_lo, y_hi = min(ys) - y_pad, max(ys) + y_pad

    ax.fill_between([0, x_hi], 0, y_lo, color="#2E7D32", alpha=0.06, zorder=0)
    ax.fill_between([x_lo, 0], 0, y_hi, color="#C62828", alpha=0.06, zorder=0)

    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="-",  zorder=1)
    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="-",  zorder=1)

    for s in scenarios:
        ax.scatter(wb_pct[s], bio_pct[s],
                   s=240 if s == baseline_name else 200,
                   color=scenario_colors[s],
                   edgecolor="#222222", linewidth=1.0,
                   marker="*" if s == baseline_name else "o",
                   label=scenario_display[s], zorder=4)
        ax.annotate(scenario_display[s],
                    xy=(wb_pct[s], bio_pct[s]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8.5, color="#222222",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                    zorder=5)

    ax.text(x_hi, y_lo, "win-win\n↑ wellbeing, ↓ impact",
            ha="right", va="bottom", fontsize=8.5,
            color="#2E7D32", fontstyle="italic", alpha=0.85)
    ax.text(x_lo, y_hi, "lose-lose\n↓ wellbeing, ↑ impact",
            ha="left", va="top", fontsize=8.5,
            color="#C62828", fontstyle="italic", alpha=0.85)
    ax.text(x_hi, y_hi, "trade-off\n↑ wellbeing, ↑ impact",
            ha="right", va="top", fontsize=8, color="#666666",
            fontstyle="italic", alpha=0.85)
    ax.text(x_lo, y_lo, "trade-off\n↓ wellbeing, ↓ impact",
            ha="left", va="bottom", fontsize=8, color="#666666",
            fontstyle="italic", alpha=0.85)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    _wb_label  = ("mean of per-indicator % change" if wellbeing_mode == "mean"
                  else "% change in summed coverage")
    _bio_label = ("% change in summed score" if impacts_mode == "sum"
                  else "geometric mean of per-category fold-change"
                  if impacts_mode == "geomean"
                  else "mean of per-category % change")

    ax.set_xlabel(
        f"Aggregated change in wellbeing across 6 DLS indicators",
        fontsize=10)
    ax.set_ylabel(
        f"Aggregated change in biodiversity impact across impact categories",
        fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(linestyle="--", linewidth=0.3, color="#CCCCCC", zorder=0)
    ax.legend(loc="best", fontsize=9, frameon=True, title="Scenario",
              title_fontsize=9)

    fig.text(
        0.5, -0.02,
        f"Y-axis ({_bio_label}): positive = greater biodiversity damage; negative = lower damage. "
        f"X-axis ({_wb_label}): positive = improved wellbeing coverage. "
        "Baseline sits at the origin.",
        ha="center", va="top", fontsize=8, style="italic", color="#555555", wrap=True,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Scenario trade-off figure saved -> {out_path}")
    return {"wellbeing_pct": wb_pct, "biodiversity_pct": bio_pct}


# ============================================================================
# Manuscript and SI figures - integrated from the former standalone
# scripts (2026-09 cleanup). Each function is self-contained: it does
# its own imports and file I/O and reads only results/tables/ CSVs, so
# the results notebook can call them in any order. Originals preserved
# in python/attic/.
# ============================================================================


def make_fig1_system_diagram():
    """Figure 1 - LCA system diagram.

    Integrated from lca_system_diagram.py.
    """
    """
    lca_system_diagram.py
    Standalone script — Figure 1: the modelled LCA system for the Tanzanian cotton
    value chain, reproducing the LCA_System sheet of cotton_industry_data.xlsx as a
    publication-ready flowchart.

    Run from the python/ directory:
        conda run -n bw25-regional python lca_system_diagram.py

    Output:
        results/figures/fig_1.png (+ .pdf)

    Structure and quantities come from the LCA_System worksheet (reference year
    2022, tonnes unless noted). Process numbers match that sheet.

    Drawing conventions
    -------------------
    * All connectors are straight (angled where they change level); nothing curves.
    * Every arrow starts and ends a fixed GAP outside the process border, computed
      by intersecting the centre-to-centre line with each box rectangle, so no head
      or tail sits on a border.
    * Elementary and technosphere inputs sit in their own colour inside a dotted
      panel, and every input arrow is exactly ARROW_LEN long regardless of whether
      it points up or down.
    * Process number badges sit inside the boxes, where no connector can cross them.
    * Waste tonnages are placed at an explicit height per stream (WASTE_LABEL_Y),
      chosen to sit clear of the oil-milling co-product arrows and the waste bus.
    * No title or subtitle: the figure carries a caption in the manuscript.
    """
    import os, sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    from src.config import RESULTS_FIGURES_DIR

    # ── Palette ───────────────────────────────────────────────────────────────────
    INK     = "#2B2B29"
    MUTED   = "#63635C"
    RULE    = "#DCDCD4"
    FARM    = "#7D9A6A"
    PROCESS = "#C08A4E"
    MARKET  = "#5A7D9A"
    WASTE   = "#8F8072"
    FLOW    = "#63635B"
    INPUT   = "#8C6D9E"    # elementary / technosphere inputs

    # ── Type scale ────────────────────────────────────────────────────────────────
    FS_BOX    = 11.4
    FS_BADGE  = 8.6
    FS_FLOW   = 10.4
    FS_TONNE  = 9.4
    FS_INPUT  = 9.3
    FS_WASTE  = 8.8
    FS_LEGEND = 10.0

    FIG_W, FIG_H = 18.5, 10.6

    BOXW, BOXH = 12.5, 6.6
    HW, HH = BOXW / 2, BOXH / 2
    GAP = 1.3              # clearance between every arrow end and a box border
    STEP = 2.95            # vertical step within an input stack
    ARROW_LEN = 2.1        # length of every input arrow, up or down
    PAD_X, PAD_Y = 1.0, 1.1    # padding inside the dotted input panel
    CHAR_W = 0.40          # approximate character width, in x-units, at FS_INPUT

    Y_TOP, Y_MID, Y_WASTE = 78.0, 47.0, 11.0
    BUS_Y = 23.0
    XS = [6.5, 28.0, 49.5, 71.0, 92.5]     # 21.5 apart: room for the flow labels

    PROCESSES = [
        ("1", "Cotton\nproduction", XS[0], Y_TOP,   FARM),
        ("2", "Ginning",            XS[1], Y_TOP,   PROCESS),
        ("3", "Spinning",           XS[2], Y_TOP,   PROCESS),
        ("4", "Weaving",            XS[3], Y_TOP,   PROCESS),
        ("5", "Dyeing",             XS[4], Y_TOP,   PROCESS),
        ("6", "Oil milling",        XS[2], Y_MID,   PROCESS),
        ("7", "Export\nmarket",     XS[0], Y_MID,   MARKET),
        ("8", "Domestic\nmarket",   XS[4], Y_MID,   MARKET),
        ("9", "Waste\nmanagement",  XS[2], Y_WASTE, WASTE),
    ]
    POS = {pid: (x, y) for pid, _, x, y, _ in PROCESSES}

    INPUTS_ABOVE = {
        "1": ["Land", "Fertiliser (N, P)", "Pesticide", "Machinery"],
        "2": ["Land", "Electricity", "Transport", "Buildings"],
        "3": ["Land", "Electricity", "Buildings"],
        "4": ["Land", "Electricity", "Maize starch", "Buildings"],
        "5": ["Land", "Water", "Electricity", "Coal", "Buildings"],
    }
    INPUTS_BELOW = {
        "6": ["Land", "Water", "Electricity", "Transport", "Buildings"],
        "7": ["Transport"],
        "8": ["Transport"],
    }
    # Nudges that keep the input panels clear of the replant loop and waste drops
    INPUT_DX = {"2": 4.0, "6": -5.0}

    CHAIN = [
        ("1", "2", "Raw cotton", 282_510),
        ("2", "3", "Lint",        20_612),
        ("3", "4", "Yarn",        18_551),
        ("4", "5", "Fabric",      18_551),
    ]
    WASTE_FLOWS = [("2", 4_294), ("3", 2_061), ("4", 1_391), ("6", 45_047)]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=200)
    fig.patch.set_facecolor("white")
    # left margin inside the axes: box borders at x=0 were getting shaved off
    # in some renderers even with savefig padding
    ax.set_xlim(-2.5, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")


    # ── Primitives ────────────────────────────────────────────────────────────────
    def rect_exit(cx, cy, dx, dy, gap=GAP):
        """Point where the ray (dx,dy) from a box centre leaves the box, plus gap."""
        n = (dx * dx + dy * dy) ** 0.5
        dx, dy = dx / n, dy / n
        tx = HW / abs(dx) if dx else float("inf")
        ty = HH / abs(dy) if dy else float("inf")
        t = min(tx, ty) + gap
        return cx + dx * t, cy + dy * t


    def arrow(p0, p1, color=FLOW, lw=1.6, style="solid", head=True, ms=13):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>" if head else "-", mutation_scale=ms,
            linewidth=lw, color=color, linestyle=style,
            shrinkA=0, shrinkB=0, zorder=2))


    def label2(x, y, name, t, color=INK, gap=1.25):
        ax.text(x, y + gap, name, fontsize=FS_FLOW, color=color, ha="center",
                va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="none"))
        ax.text(x, y - gap, f"{t:,.0f} t", fontsize=FS_TONNE, color=MUTED,
                ha="center", va="top", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none"))


    def connect(a, b, label=None, t=None, loff=(0.0, 0.0)):
        xa, ya = POS[a]
        xb, yb = POS[b]
        p0 = rect_exit(xa, ya, xb - xa, yb - ya)
        p1 = rect_exit(xb, yb, xa - xb, ya - yb)
        arrow(p0, p1)
        if label:
            label2((p0[0] + p1[0]) / 2 + loff[0], (p0[1] + p1[1]) / 2 + loff[1],
                   label, t)


    def box(pid, label, x, y, color):
        ax.add_patch(FancyBboxPatch(
            (x - HW, y - HH), BOXW, BOXH,
            boxstyle="round,pad=0.35,rounding_size=1.1",
            linewidth=1.8, edgecolor=color, facecolor=color + "1A", zorder=3))
        ax.text(x, y - 0.6, label, fontsize=FS_BOX, fontweight="bold", color=INK,
                ha="center", va="center", zorder=4, linespacing=1.2)
        ax.add_patch(plt.Circle((x - HW + 2.0, y + HH - 1.6), 1.2,
                                facecolor=color, edgecolor="none", zorder=5))
        ax.text(x - HW + 2.0, y + HH - 1.6, pid, fontsize=FS_BADGE,
                fontweight="bold", color="white", ha="center", va="center", zorder=6)


    def input_stack(pid, items, below=False):
        """Dotted panel of inputs, joined to the process by a fixed-length arrow."""
        x, y = POS[pid]
        x += INPUT_DX.get(pid, 0.0)
        h_text = (len(items) - 1) * STEP
        w = max(len(s) for s in items) * CHAR_W + 2 * PAD_X
        h = h_text + 2 * PAD_Y

        if below:
            panel_top = y - HH - GAP - ARROW_LEN
            y_first = panel_top - PAD_Y
            panel_bottom = panel_top - h
            arrow((x, panel_top), (x, y - HH - GAP), color=INPUT, lw=1.2, ms=10)
        else:
            panel_bottom = y + HH + GAP + ARROW_LEN
            y_first = panel_bottom + PAD_Y + h_text
            arrow((x, panel_bottom), (x, y + HH + GAP), color=INPUT, lw=1.2, ms=10)

        ax.add_patch(FancyBboxPatch(
            (x - w / 2, panel_bottom), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.8",
            linewidth=1.0, edgecolor=INPUT, facecolor="white",
            linestyle=(0, (1.6, 1.8)), zorder=3))

        for i, it in enumerate(items):
            ax.text(x, y_first - i * STEP, it, fontsize=FS_INPUT, color=INPUT,
                    ha="center", va="center", zorder=4)


    # ── Draw ──────────────────────────────────────────────────────────────────────
    for pid, items in INPUTS_ABOVE.items():
        input_stack(pid, items)
    for pid, items in INPUTS_BELOW.items():
        input_stack(pid, items, below=True)

    for pid, label, px, py, pcolor in PROCESSES:
        box(pid, label, px, py, pcolor)

    for a, b, name, t in CHAIN:
        connect(a, b, name, t)

    connect("2", "7", "Lint", 82_448, loff=(-5.2, 0.0))
    connect("2", "6", "Seed", 150_156, loff=(5.2, 0.0))
    connect("5", "8", "Textile", 17_159, loff=(7.4, 0.0))

    x6, _ = POS["6"]
    x8, _ = POS["8"]
    for dy, name, t in [(1.9, "Seed oil", 24_776), (-1.9, "Seed cake", 80_333)]:
        arrow((x6 + HW + GAP, Y_MID + dy), (x8 - HW - GAP, Y_MID + dy))
        label2(78.0, Y_MID + dy + (3.8 if dy > 0 else -3.8), name, t, gap=1.1)

    # Retained planting seed: orthogonal loop clearing both input panels
    Y_LOOP = 97.0
    LX0, LX1 = 24.5, 11.8
    ax.plot([LX0, LX0], [Y_TOP + HH + 2.6, Y_LOOP], color=FARM, linewidth=1.4,
            linestyle=(0, (4, 2.6)), zorder=2)
    ax.plot([LX0, LX1], [Y_LOOP, Y_LOOP], color=FARM, linewidth=1.4,
            linestyle=(0, (4, 2.6)), zorder=2)
    arrow((LX1, Y_LOOP), (LX1, Y_TOP + HH + 2.6), color=FARM, lw=1.4,
          style=(0, (4, 2.6)), ms=12)
    ax.text((LX0 + LX1) / 2, Y_LOOP + 1.4, "Seed retained for replanting",
            fontsize=FS_FLOW - 0.6, color=FARM, ha="center", va="bottom", zorder=6)
    ax.text((LX0 + LX1) / 2, Y_LOOP - 1.5, "25,000 t", fontsize=FS_TONNE - 0.4,
            color=FARM, ha="center", va="top", zorder=6)

    # ── Waste bus ─────────────────────────────────────────────────────────────────
    WSTYLE = (0, (3.2, 2.4))
    drops = {"2": 28.0, "4": 71.0, "6": 53.0}
    SPIN_X, SPIN_ELBOW_Y, SPIN_DROP_X = 46.5, 69.0, 61.5
    WASTE_LABEL_Y = {"2": 66.0, "3": 60.0, "4": 66.0, "6": 34.0}


    def waste_label(x, pid, t):
        ax.text(x + 1.1, WASTE_LABEL_Y[pid], f"{t:,.0f} t", fontsize=FS_WASTE,
                color=WASTE, ha="left", va="center", rotation=90, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none"))


    for pid, t in WASTE_FLOWS:
        if pid == "3":
            continue
        xp, yp = POS[pid]
        xd = drops[pid]
        ax.plot([xd, xd], [yp - HH - GAP, BUS_Y], color=WASTE, linewidth=1.2,
                linestyle=WSTYLE, zorder=2)
        waste_label(xd, pid, t)

    # Spinning: elbowed so the line leaves the box itself, then clears oil milling
    _, y3 = POS["3"]
    ax.plot([SPIN_X, SPIN_X], [y3 - HH - GAP, SPIN_ELBOW_Y], color=WASTE,
            linewidth=1.2, linestyle=WSTYLE, zorder=2)
    ax.plot([SPIN_X, SPIN_DROP_X], [SPIN_ELBOW_Y, SPIN_ELBOW_Y], color=WASTE,
            linewidth=1.2, linestyle=WSTYLE, zorder=2)
    ax.plot([SPIN_DROP_X, SPIN_DROP_X], [SPIN_ELBOW_Y, BUS_Y], color=WASTE,
            linewidth=1.2, linestyle=WSTYLE, zorder=2)
    waste_label(SPIN_DROP_X, "3", 2_061)

    xs_all = list(drops.values()) + [SPIN_DROP_X]
    ax.plot([min(xs_all), max(xs_all)], [BUS_Y, BUS_Y], color=WASTE,
            linewidth=1.2, linestyle=WSTYLE, zorder=2)
    arrow((XS[2], BUS_Y), (XS[2], Y_WASTE + HH + GAP), color=WASTE, lw=1.4,
          style=WSTYLE, ms=13)

    # ── Legend ────────────────────────────────────────────────────────────────────
    LEG_Y = 2.6
    ax.plot([0.5, 99.5], [5.4, 5.4], color=RULE, linewidth=1.0, zorder=0)
    for i, (c, lbl) in enumerate([(FARM, "Cultivation"), (PROCESS, "Processing"),
                                  (MARKET, "Final market"),
                                  (WASTE, "Waste management")]):
        xx = 0.5 + i * 14.0
        ax.add_patch(FancyBboxPatch((xx, LEG_Y - 0.85), 2.8, 1.8,
                                    boxstyle="round,pad=0.1,rounding_size=0.4",
                                    linewidth=1.4, edgecolor=c,
                                    facecolor=c + "1A", zorder=3))
        ax.text(xx + 3.9, LEG_Y, lbl, fontsize=FS_LEGEND, color=MUTED,
                ha="left", va="center")

    ax.add_patch(FancyBboxPatch((58.0, LEG_Y - 0.85), 2.8, 1.8,
                                boxstyle="round,pad=0.1,rounding_size=0.4",
                                linewidth=1.0, edgecolor=INPUT, facecolor="white",
                                linestyle=(0, (1.6, 1.8)), zorder=3))
    ax.text(61.9, LEG_Y, "Elementary and technosphere inputs",
            fontsize=FS_LEGEND, color=INPUT, ha="left", va="center")

    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    for ext in ("png",):
        out = os.path.join(RESULTS_FIGURES_DIR, f"fig_1.{ext}")
        # pad so box borders at the canvas edge are not shaved off in print
        plt.savefig(out, dpi=600, facecolor="white", bbox_inches="tight",
                    pad_inches=0.15)
        print(f"Saved -> {out}")
    plt.close()


def make_fig2_scenario_propagation():
    """Figure 2 - scenario -> description -> model inputs changed.

    Integrated from scenario_propagation.py.
    """
    """
    scenario_propagation.py
    Standalone script — Figure 2: how each scenario lever propagates through the
    model into changed model inputs, and from there into affected outcomes.

    Reads pre-computed CSVs from results/tables/. No Brightway needed.
    Run from the python/ directory:
        conda run -n bw25-regional python scenario_propagation.py

    Output:
        results/figures/fig_2.png (+ .pdf)

    Design notes
    ------------
    * The middle column describes WHAT THE SCENARIO CHANGES in the model. It is
      deliberately free of up/down arrows and fold-factors: those belong to the
      outcomes column, and mixing them made the two columns hard to tell apart.
    * The outcomes column carries the arrows and fold-changes, colour-coded by
      whether a change is an environmental burden, a human benefit, or an
      invariant held fixed by the scenario design.
    * Fold-changes are READ FROM THE RESULTS TABLES rather than hardcoded, so the
      figure stays correct after the LCA is re-run.
    * Outcomes are named for the DRIVER, not the impact category: "nitrogen /
      phosphorus application" rather than "eutrophication", "pesticides" rather
      than "ecotoxicity", "emissions" rather than "climate". Each of these
      fold-changes is the underlying flow's fold-change passed through a linear
      characterisation factor, so naming the impact category would imply a
      computed damage result the figure is not showing.
    """
    import sys, os

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Circle

    from src.config import RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR

    # ── Data ──────────────────────────────────────────────────────────────────────
    _imp = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "all_scenarios.csv"))
    _dis = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "district_scores_all_scenarios.csv"))
    _dls = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "dls_coverage_all_scenarios.csv"))

    _piv = _imp.pivot_table(index="category", columns="scenario", values="score", aggfunc="first")
    # Water: exclude the Chemba (TZ0107) basin-CF outlier, as every other figure does
    _piv.loc["Water"] = (
        _dis[(_dis.category == "Water consumption") & (~_dis.district_id.isin({"TZ0107"}))]
        .groupby("scenario")["score"].sum()
    )
    _dls["indicator"] = _dls["indicator"].str.replace("\n", " ", regex=False)
    _dpiv = _dls.pivot_table(index="indicator", columns="scenario", values="coverage", aggfunc="first")


    def fc(category, scen):
        """Impact fold-change vs baseline."""
        return _piv.loc[category, scen] / _piv.loc[category, "baseline"]


    def dfc(indicator, scen):
        """DLS coverage fold-change vs baseline."""
        return _dpiv.loc[indicator, scen] / _dpiv.loc[indicator, "baseline"]


    def x(v):
        """Format a fold-change compactly."""
        return f"{v:,.0f}×" if v >= 100 else f"{v:.1f}×"


    # ── Palette ───────────────────────────────────────────────────────────────────
    INK     = "#24242B"
    MUTED   = "#6A6A72"
    FAINT   = "#B4B4BC"
    RULE    = "#DEDEE4"
    BURDEN  = "#D1495B"   # environmental burden
    BENEFIT = "#0F8B8D"   # human benefit
    NEUTRAL = "#A9A9B2"   # unchanged by design

    SCEN_COLOR = {
        "high_yield":              "#1F77B4",
        "extensification":         "#8C564B",
        "organic_expansion":       "#2CA02C",
        "manufacturing_expansion": "#9467BD",
        "irrigation":              "#FF7F0E",
    }

    # ── Content ───────────────────────────────────────────────────────────────────
    # (key, display label, [model-input lines], [(colour, outcome text)])
    ROWS = [
        ("high_yield", "High yield",
         ["Yield raised to national target of 2,471 kg/ha.",
          "Per-kilogram agrochemical rates unchanged.",
          "Land and tractor use per kg fall."],
         [(BURDEN,  "Nitrogen, phosphorus application ↑ "
                    f"{x(fc('FW eutrophication N', 'high_yield'))}"),
          (BURDEN,  f"Emissions ↑ {x(fc('Climate change rcp26', 'high_yield'))}"
                    f"  ·  pesticides ↑ {x(fc('FW ecotoxicity', 'high_yield'))}"),
          (BENEFIT, f"Farm income ↑ {x(dfc('Farm income (living-income)', 'high_yield'))}"
                    f"  ·  nutrition ↑ {x(dfc('Nutrition (kcal)', 'high_yield'))}"),
          (BENEFIT, f"Employment ↑ {x(dfc('Employment (rural labour)', 'high_yield'))}"),
          (NEUTRAL, "Land use unchanged")]),

        ("extensification", "Extensification",
         ["Cultivated area expanded to match total",
          "output from the 'high yield' scenario.",
          "All per-hectare rates remain unchanged."],
         [(BURDEN,  f"Land use ↑ {x(fc('Land use occupation', 'extensification'))}"),
          (BURDEN,  "Nitrogen, phosphorus application ↑ "
                    f"{x(fc('FW eutrophication N', 'extensification'))}"),
          (BURDEN,  f"Emissions ↑ {x(fc('Climate change rcp26', 'extensification'))}"
                    f"  ·  pesticides ↑ {x(fc('FW ecotoxicity', 'extensification'))}"),
          (BENEFIT, f"Nutrition ↑ {x(dfc('Nutrition (kcal)', 'extensification'))}"
                    f"  ·  employment ↑ {x(dfc('Employment (rural labour)', 'extensification'))}"),
          (NEUTRAL, "Farm income unchanged")]),

        ("organic_expansion", "Organic expansion",
         ["Organic share of cultivated area doubled.",
          "Manure substitutes for urea; pesticide use falls.",
          "Farmgate price carries an organic premium."],
         [(BURDEN,  f"Nitrogen application ↑ {x(fc('FW eutrophication N', 'organic_expansion'))}"
                    f"  ·  phosphorus ↑ {x(fc('FW eutrophication P', 'organic_expansion'))}"),
          (BENEFIT, f"Farm income ↑ {x(dfc('Farm income (living-income)', 'organic_expansion'))}"
                    "  (price premium)"),
          (NEUTRAL, "Emissions, pesticides and land use unchanged")]),

        ("manufacturing_expansion", "Manufacturing expansion",
         ["Domestic fabric output tripled.",
          "More lint retained for domestic processing.",
          "No farm-stage parameters change."],
         [(BURDEN,  f"Water ↑ {x(fc('Water', 'manufacturing_expansion'))}"
                    f"  ·  pesticides ↑ {x(fc('FW ecotoxicity', 'manufacturing_expansion'))}"),
          (BENEFIT, f"Clothing ↑ {x(dfc('Clothing', 'manufacturing_expansion'))}"),
          (NEUTRAL, "Farm-stage impacts unchanged")]),

        ("irrigation", "Irrigation",
         ["Yield raised through supplemental irrigation.",
          "Irrigation water added as a new input.",
          "Per-hectare field inputs held at baseline."],
         [(BURDEN,  f"Water ↑ {x(fc('Water', 'irrigation'))}"),
          (BURDEN,  f"Emissions ↑ {x(fc('Climate change rcp26', 'irrigation'))}"
                    f"  ·  pesticides ↑ {x(fc('FW ecotoxicity', 'irrigation'))}"),
          (BENEFIT, f"Farm income ↑ {x(dfc('Farm income (living-income)', 'irrigation'))}"
                    f"  ·  nutrition ↑ {x(dfc('Nutrition (kcal)', 'irrigation'))}"),
          (NEUTRAL, "Nitrogen, phosphorus application and land use unchanged")]),
    ]

    # ── Geometry ──────────────────────────────────────────────────────────────────
    # Laid out in INCHES, then converted to figure fractions, so line spacing and
    # row padding stay visually constant regardless of how many rows there are.
    X_L, X_R = 0.030, 0.970           # page margins
    X_BOX_L, X_BOX_R = 0.030, 0.205   # scenario box
    X_MID = 0.280                     # description text left edge
    X_DOT = 0.612                     # outcome dot centre
    X_OUT = 0.628                     # outcomes text left edge

    LINE_H_IN  = 0.265    # vertical step between text lines
    ROW_PAD_IN = 0.34     # padding above + below a row's text block
    GAP_IN     = 0.40     # space between rows (the dotted rule sits mid-gap)
    TITLE_IN   = 1.45     # title + subtitle + column headers + rule
    LEGEND_IN  = 0.90     # rule + legend strip

    n_lines  = [max(len(m), len(o)) for _, _, m, o in ROWS]
    row_h_in = [n * LINE_H_IN + ROW_PAD_IN for n in n_lines]

    FIG_W = 12.0
    FIG_H = TITLE_IN + sum(row_h_in) + GAP_IN * (len(ROWS) - 1) + LEGEND_IN

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def fy(inches_from_top):
        """Figure-fraction y from a distance in inches below the top edge."""
        return 1.0 - inches_from_top / FIG_H

    step = LINE_H_IN / FIG_H          # line spacing, in figure fraction

    # ── Title block ───────────────────────────────────────────────────────────────
    fig.text(X_L, fy(0.40), "How each scenario propagates through the model",
             fontsize=16, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(X_L, fy(0.76),
             "Scenario  →  description  →  model inputs changed",
             fontsize=10.2, color=MUTED, va="top", ha="left")

    hdr_y = fy(1.16)
    for xx, lbl in [(X_BOX_L, "Scenario"), (X_MID, "Description"),
                    (X_DOT - 0.016, "Model inputs changed")]:
        fig.text(xx, hdr_y, lbl.upper(), fontsize=8.4, color=MUTED,
                 va="center", ha="left", fontweight="bold")

    rule_y = fy(1.34)
    ax.plot([X_L, X_R], [rule_y, rule_y], color=RULE, linewidth=1.0, zorder=1)

    # ── Rows ──────────────────────────────────────────────────────────────────────
    cursor_in = TITLE_IN
    for i, ((key, label, mid_lines, outcomes), h_in) in enumerate(zip(ROWS, row_h_in)):
        y_mid = fy(cursor_in + h_in / 2)
        cursor_in += h_in

        color = SCEN_COLOR[key]

        box_h = 0.46 / FIG_H
        ax.add_patch(FancyBboxPatch(
            (X_BOX_L, y_mid - box_h / 2), X_BOX_R - X_BOX_L, box_h,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=1.8, edgecolor=color, facecolor="white", zorder=3))
        ax.text((X_BOX_L + X_BOX_R) / 2, y_mid, label, fontsize=10.4,
                fontweight="bold", color=INK, ha="center", va="center", zorder=4)

        y0 = y_mid + step * (len(mid_lines) - 1) / 2
        for j, ln in enumerate(mid_lines):
            ax.text(X_MID, y0 - j * step, ln, fontsize=9.5, color=INK,
                    ha="left", va="center", zorder=4)

        y0 = y_mid + step * (len(outcomes) - 1) / 2
        for j, (c, txt) in enumerate(outcomes):
            yy = y0 - j * step
            ax.add_patch(Circle((X_DOT, yy), 0.0038, facecolor=c,
                                edgecolor="none", zorder=4))
            ax.text(X_OUT, yy, txt, fontsize=9.5,
                    color=MUTED if c == NEUTRAL else INK,
                    ha="left", va="center", zorder=4)

        if i < len(ROWS) - 1:
            # small centred solid bar separating scenarios
            y_sep = fy(cursor_in + GAP_IN / 2)
            ax.plot([0.46, 0.54], [y_sep, y_sep], linewidth=1.6, color=FAINT,
                    solid_capstyle="round", zorder=1)
            cursor_in += GAP_IN

    # ── Legend ────────────────────────────────────────────────────────────────────
    leg_y = fy(cursor_in + 0.52)
    ax.plot([X_L, X_R], [fy(cursor_in + 0.18), fy(cursor_in + 0.18)],
            color=RULE, linewidth=1.0, zorder=1)

    for xx, c, lbl in [(X_L, BURDEN, "environmental burden"),
                       (X_L + 0.205, BENEFIT, "human benefit"),
                       (X_L + 0.365, NEUTRAL, "unchanged")]:
        ax.add_patch(Circle((xx + 0.005, leg_y), 0.0040, facecolor=c,
                            edgecolor="none", zorder=4))
        ax.text(xx + 0.018, leg_y, lbl, fontsize=8.9, color=MUTED,
                ha="left", va="center")

    fig.text(X_R, leg_y, "Fold-changes are relative to the baseline scenario.",
             fontsize=8.5, color=FAINT, ha="right", va="center", style="italic")

    # ── Save ──────────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    for ext in ("png",):
        out = os.path.join(RESULTS_FIGURES_DIR, f"fig_2.{ext}")
        plt.savefig(out, dpi=600, facecolor="white", bbox_inches="tight")
        print(f"Saved -> {out}")
    plt.close()


def make_fig2b_scenario_bubbles():
    """Figure 2b (exploratory) - scenario x outcome bubble matrix.

    Integrated from fig2b_scenario_bubbles.py.
    """
    """
    fig2b_scenario_bubbles.py
    Standalone script — Figure 2b: bubble-matrix companion to the scenario
    propagation figure. One row per scenario, one column per outcome (six
    environmental pressures + five DLS indicators); bubble area encodes the
    magnitude of change vs baseline.

    Reads pre-computed CSVs from results/tables/. No Brightway needed.
    Run from the python/ directory:
        conda run -n bw25-regional python fig2b_scenario_bubbles.py

    Output:
        results/figures/fig_2_b.png (+ .pdf)

    Design notes
    ------------
    * Column labels follow fig2's convention: outcomes are named for the DRIVER
      ("pesticides", "emissions", "nitrogen application"), not the impact
      category, because each fold-change is the underlying flow's change passed
      through a linear CF.
    * Bubble area is linear in % change vs baseline, capped at +300% (the bulk
      of the data); irrigation's water bubble (+19,700% off a near-zero
      rain-fed baseline) is drawn at the cap with a white break slash, matching
      fig5's treatment of the same outlier.
    * Unchanged cells (|change| < 0.5%) get a small open grey circle — the
      "unchanged by design" state from fig2's legend, kept visible so absence
      of burden reads as information, not as missing data.
    * Colours are fig2's: burden red, benefit teal, neutral grey.
    """
    import sys, os

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.config import RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR

    # ── Data (same conventions as scenario_propagation.py) ───────────────────────
    _imp = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "all_scenarios.csv"))
    _dis = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "district_scores_all_scenarios.csv"))
    _dls = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "dls_coverage_all_scenarios.csv"))

    _piv = _imp.pivot_table(index="category", columns="scenario", values="score", aggfunc="first")
    # Water: exclude the Chemba (TZ0107) basin-CF outlier, as every other figure does
    _piv.loc["Water"] = (
        _dis[(_dis.category == "Water consumption") & (~_dis.district_id.isin({"TZ0107"}))]
        .groupby("scenario")["score"].sum()
    )
    _dls["indicator"] = _dls["indicator"].str.replace("\n", " ", regex=False)
    _dpiv = _dls.pivot_table(index="indicator", columns="scenario", values="coverage", aggfunc="first")

    # ── Palette (fig2's) ─────────────────────────────────────────────────────────
    INK, MUTED, FAINT, RULE = "#24242B", "#6A6A72", "#B4B4BC", "#DEDEE4"
    BURDEN, BENEFIT, NEUTRAL = "#D1495B", "#0F8B8D", "#A9A9B2"
    SCEN_COLOR = {
        "high_yield":              "#1F77B4",
        "extensification":         "#8C564B",
        "organic_expansion":       "#2CA02C",
        "manufacturing_expansion": "#9467BD",
        "irrigation":              "#FF7F0E",
    }
    SCEN = [("high_yield", "High yield"), ("extensification", "Extensification"),
            ("organic_expansion", "Organic expansion"),
            ("manufacturing_expansion", "Manufacturing expansion"),
            ("irrigation", "Irrigation")]

    # Columns: fig2 driver naming. (label, kind, table-key)
    COLS = [
        ("Land use",               "imp", "Land use occupation"),
        ("Nitrogen\napplication",   "imp", "FW eutrophication N"),
        ("Phosphorus\napplication", "imp", "FW eutrophication P"),
        ("Emissions",              "imp", "Climate change rcp26"),
        ("Pesticides",             "imp", "FW ecotoxicity"),
        ("Water",                  "imp", "Water"),
        ("Farm\nincome",           "dls", "Farm income (living-income)"),
        ("Clothing",               "dls", "Clothing"),
        ("Employment",             "dls", "Employment (rural labour)"),
        ("Nutrition\n(kcal)",      "dls", "Nutrition (kcal)"),
        ("Nutrition\n(protein)",   "dls", "Nutrition (protein)"),
    ]
    N_IMP = sum(1 for _, k, _k in COLS if k == "imp")

    CAP_PCT = 300.0     # bubble scale set by the bulk of the data
    SMAX    = 1350.0    # points^2 at the cap
    THRESH  = 1.0       # |% change| below this -> "unchanged" open circle
                        # (1.0 keeps organic's -0.5% pesticide change in the
                        # "unchanged" state, matching fig2's designation)


    def pct_change(kind, key, scen):
        tab = _piv if kind == "imp" else _dpiv
        return 100.0 * (tab.loc[key, scen] / tab.loc[key, "baseline"] - 1.0)


    # ── Figure ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12.0, 5.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(-0.7, len(COLS) - 0.3)
    ax.set_ylim(-0.75, len(SCEN) - 0.25)
    ax.invert_yaxis()
    ax.axis("off")

    # group headers, fig2 header style
    ax.text((N_IMP - 1) / 2, -0.62, "ENVIRONMENTAL BURDENS", fontsize=8.6,
            color=MUTED, fontweight="bold", ha="center", va="center")
    ax.text(N_IMP + (len(COLS) - N_IMP - 1) / 2, -0.62, "HUMAN BENEFITS (DLS)",
            fontsize=8.6, color=MUTED, fontweight="bold", ha="center", va="center")

    # divider between the two column groups
    ax.plot([N_IMP - 0.5, N_IMP - 0.5], [-0.55, len(SCEN) - 0.55],
            color=RULE, lw=1.0, zorder=1)

    # faint row rules
    for r in range(len(SCEN)):
        ax.plot([-0.55, len(COLS) - 0.45], [r, r], color=RULE, lw=0.6,
                linestyle=(0, (1, 3.4)), zorder=1)

    # column labels along the bottom
    for c, (lbl, kind, key) in enumerate(COLS):
        ax.text(c, len(SCEN) - 0.52, lbl, fontsize=8.8, color=INK,
                ha="center", va="top")

    # scenario labels at the left, in scenario colours (fig2 box colours)
    for r, (skey, slabel) in enumerate(SCEN):
        ax.text(-0.68, r, slabel, fontsize=10.0, fontweight="bold",
                color=SCEN_COLOR[skey], ha="right", va="center", clip_on=False)

    # bubbles
    for r, (skey, _) in enumerate(SCEN):
        for c, (lbl, kind, key) in enumerate(COLS):
            p = pct_change(kind, key, skey)
            if abs(p) < THRESH:
                ax.scatter(c, r, s=34, facecolor="white", edgecolor=NEUTRAL,
                           linewidth=1.1, zorder=3)
                continue
            capped = p > CAP_PCT
            s = min(abs(p), CAP_PCT) / CAP_PCT * SMAX
            color = BURDEN if kind == "imp" else BENEFIT
            if p < 0:      # a decrease: open bubble in the same hue
                ax.scatter(c, r, s=s, facecolor="white", edgecolor=color,
                           linewidth=1.6, zorder=3)
            else:
                ax.scatter(c, r, s=s, facecolor=color, edgecolor="white",
                           linewidth=1.2, alpha=0.92, zorder=3)
            if capped:     # off-scale outlier: white break slash, as in fig5
                ax.plot([c - 0.22, c + 0.22], [r + 0.10, r - 0.02],
                        color="white", lw=2.2, zorder=4)
                ax.plot([c - 0.22, c + 0.22], [r + 0.16, r + 0.04],
                        color="white", lw=2.2, zorder=4)

    # ── Legend: bubble sizes + states ────────────────────────────────────────────
    ly = len(SCEN) + 0.32
    ax.set_ylim(ly + 0.62, -0.75)          # extend downward (axis inverted)
    for xx, pv in [(0.0, 50), (1.15, 150), (2.3, 300)]:
        ax.scatter(xx, ly, s=pv / CAP_PCT * SMAX, facecolor="none",
                   edgecolor=MUTED, linewidth=1.1, zorder=3, clip_on=False)
        ax.text(xx, ly + 0.56, f"+{pv}%", fontsize=8.0, color=MUTED,
                ha="center", va="center", clip_on=False)
    ax.text(3.1, ly, "change vs baseline (bubble area)", fontsize=8.6,
            color=MUTED, ha="left", va="center")

    lx = 6.4
    ax.scatter(lx, ly, s=170, facecolor=BURDEN, edgecolor="white",
               linewidth=1.0, alpha=0.92, clip_on=False)
    ax.text(lx + 0.25, ly, "burden ↑", fontsize=8.6, color=MUTED,
            ha="left", va="center")
    ax.scatter(lx + 1.6, ly, s=170, facecolor=BENEFIT, edgecolor="white",
               linewidth=1.0, alpha=0.92, clip_on=False)
    ax.text(lx + 1.85, ly, "benefit ↑", fontsize=8.6, color=MUTED,
            ha="left", va="center")
    ax.scatter(lx + 3.2, ly, s=34, facecolor="white", edgecolor=NEUTRAL,
               linewidth=1.1, clip_on=False)
    ax.text(lx + 3.45, ly, "unchanged", fontsize=8.6, color=MUTED,
            ha="left", va="center")
    ax.text(len(COLS) - 0.35, ly + 0.42,
            "slashed bubble: off scale (water ×198)", fontsize=7.8,
            color=FAINT, ha="right", va="center", style="italic", clip_on=False)

    # ── Save ─────────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    for ext in ("png",):
        out = os.path.join(RESULTS_FIGURES_DIR, f"fig_2_b.{ext}")
        plt.savefig(out, dpi=600, facecolor="white", bbox_inches="tight",
                    pad_inches=0.12)
        print(f"Saved -> {out}")
    plt.close()

    print(f"\n{'':24s}" + "".join(f"{l.replace(chr(10), ' '):>14s}" for l, _, _ in COLS))
    for skey, slabel in SCEN:
        row = "".join(f"{pct_change(k, key, skey):>+14.1f}"
                      for _, k, key in COLS)
        print(f"{slabel:<24s}{row}")


def make_fig3_disaggregation():
    """Figure 3 - combined impact disaggregation maps (shared scale).

    Integrated from fig3_impact_disaggregation.py.
    """
    """
    fig3_impact_disaggregation.py
    Standalone script — Figure 3 (version A, SHARED colour scale): how the combined
    regionalised biodiversity impact of the baseline scenario disaggregates into its
    four spatial categories.

    Run from the python/ directory:
        conda run -n bw25-regional python fig3_impact_disaggregation.py

    Output:
        results/figures/fig_3.png (+ .pdf)

    See fig3b_impact_disaggregation_free.py for the companion version in which every
    panel carries its own colour scale.

    Reads results/tables/district_scores_all_scenarios.csv. No Brightway needed.

    Design notes
    ------------
    * Values are ABSOLUTE, in PDF*yr, and the combined panel is a plain sum of the
      four categories. No normalisation. All four categories are already in PDF*yr:
      the land-use score works out to ~2.6e-15 PDF*yr per m2*yr of occupation,
      which is the right order for LC-IMPACT regionalised occupation CFs, so the
      "PDF*m2*year" string in the method metadata describes the FLOW, not the
      score. An earlier version normalised each category to its national total,
      which silently gave water — 0.001% of baseline damage — the same weight as
      land use, which is 99.99% of it.
    * One shared logarithmic colour scale across all five panels. That is the whole
      point of this version: it shows honestly that combined regionalised impact is
      almost entirely land use, and that water and eutrophication sit several
      orders of magnitude below. The scale spans 1e-13 to 1e-5 PDF*yr.
    * Districts absent from a category are drawn in neutral grey rather than as
      zero. Baseline cotton is rainfed, so water consumption occurs in only six
      districts — the ginning and textile sites — and that sparsity is a result.
    * NOTE for the caption: these are the four REGIONALISED categories only.
      Climate change and freshwater ecotoxicity have no sub-national CFs and are
      absent, so "combined" here is not total damage.
    * Chemba (TZ0107) is excluded from water, as in every other figure.
    """
    import os, sys

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    gpd.options.io_engine = "fiona"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    from src.config import TZ_DISTRICTS_PATH, RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR

    SCENARIO = "baseline"
    WATER_EXCLUDE = {"TZ0107"}
    N_LABEL = 4                     # districts labelled on the combined panel

    INK   = "#2B2B29"
    MUTED = "#63635C"
    FAINT = "#9A9A92"
    BASE  = "#EAEAE4"
    EDGE  = "#FFFFFF"
    OUTL  = "#8A8A82"
    CMAP  = "magma_r"

    VMIN, VMAX = 1e-13, 1e-5        # PDF*yr

    CATS = [
        ("Land use occupation", "Land use",           "#6E8C5A"),
        ("Water consumption",   "Water consumption",  "#4E7291"),
        ("FW eutrophication N", "Eutrophication (N)", "#C08A4E"),
        ("FW eutrophication P", "Eutrophication (P)", "#8E6086"),
    ]

    # ── Data ──────────────────────────────────────────────────────────────────────
    gdf = gpd.read_file(TZ_DISTRICTS_PATH)
    outline = gdf.dissolve().boundary

    d = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "district_scores_all_scenarios.csv"))
    d = d[~((d.district_id.isin(WATER_EXCLUDE)) & (d.category == "Water consumption"))]
    d = d[d.scenario == SCENARIO]

    combined = d.groupby("district_id")["score"].sum()
    panels = [("a) Total", combined, None, INK)]
    _letters = "bcde"
    for (cat, label, colour), lt in zip(CATS, _letters):
        panels.append((f"{lt}) {label}",
                       d[d.category == cat].set_index("district_id")["score"],
                       cat, colour))

    minx, miny, maxx, maxy = gdf.total_bounds
    mx, my = (maxx - minx) * 0.02, (maxy - miny) * 0.02
    XLIM, YLIM = (minx - mx, maxx + mx), (miny - my, maxy + my)

    # ── Figure ────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16.0, 8.6), dpi=200)
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 4, figure=fig, hspace=0.10, wspace=0.03,
                  left=0.022, right=0.978, top=0.945, bottom=0.135)

    norm = LogNorm(vmin=VMIN, vmax=VMAX, clip=True)
    cmap = matplotlib.colormaps.get_cmap(CMAP)
    name_by_code = dict(zip(gdf.ADM2_PCODE, gdf.ADM2_EN))


    def draw(ax, series, title, accent, big=False):
        g = gdf.copy()
        g["v"] = g["ADM2_PCODE"].map(series)
        has = g["v"].notna() & (g["v"] > 0)

        g[~has].plot(ax=ax, facecolor=BASE, edgecolor=EDGE, linewidth=0.22, zorder=1)
        g[has].plot(ax=ax, column="v", cmap=cmap, norm=norm,
                    edgecolor=EDGE, linewidth=0.22, zorder=2)
        outline.plot(ax=ax, color=OUTL, linewidth=0.7, zorder=3)

        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.set_axis_off()

        ax.set_title(title, fontsize=13.5 if big else 11.0, fontweight="bold",
                     color=INK, pad=10 if big else 7)

        # a handful of tiny districts can be hard to find; ring them
        if 0 < has.sum() <= 10:
            for geom in g.loc[has, "geometry"]:
                c = geom.representative_point()
                ax.plot(c.x, c.y, marker="o", markersize=9, markerfacecolor="none",
                        markeredgecolor=accent, markeredgewidth=1.5, zorder=6)
        return ax


    ax_tot = fig.add_subplot(gs[:, 0:2])
    lbl, s_tot, _, col = panels[0]
    draw(ax_tot, s_tot, lbl, col, big=True)

    # ── Label the largest districts ───────────────────────────────────────────────
    # The top districts sit in one tight northern cluster, so labels must be pushed
    # well clear of it. For each district, candidate positions fan out at a range of
    # angles and radii; the first candidate that clears the map edge, every label
    # already placed, AND every leader already drawn wins. Leaders therefore radiate
    # in different directions rather than stacking into a column of parallel
    # vertical lines on top of one another.
    LBL_W, LBL_H = 2.2, 1.35          # approximate label extent, in degrees
    _ANGLES = [0, -30, 30, -60, 60, -90, 90, -125, 125, 180, -155, 155]
    _RADII = (3.4, 5.0, 6.8)


    def _boxes_clear(lx, ly, boxes):
        return not any(abs(lx - px) < LBL_W and abs(ly - py) < LBL_H
                       for px, py in boxes)


    def _leader_clear(cx, cy, lx, ly, boxes):
        """True if the leader line does not pass under an already-placed label."""
        for t in np.linspace(0.0, 1.0, 16):
            sx, sy = cx + t * (lx - cx), cy + t * (ly - cy)
            if any(abs(sx - px) < LBL_W * 0.62 and abs(sy - py) < LBL_H * 0.62
                   for px, py in boxes):
                return False
        return True


    def _crosses(p1, p2, segs):
        """True if segment p1-p2 crosses any leader already drawn."""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        for p3, p4 in segs:
            if (ccw(p1, p3, p4) != ccw(p2, p3, p4)
                    and ccw(p1, p2, p3) != ccw(p1, p2, p4)):
                return True
        return False


    def label_top_districts(ax, series, n):
        boxes, segs = [], []
        for code, val in series.sort_values(ascending=False).head(n).items():
            geom = gdf.loc[gdf.ADM2_PCODE == code, "geometry"]
            if geom.empty:
                continue
            c = geom.iloc[0].representative_point()

            pick = None
            for r in _RADII:
                for a in _ANGLES:
                    lx = c.x + r * np.cos(np.radians(a))
                    ly = c.y + r * np.sin(np.radians(a))
                    if not (XLIM[0] + LBL_W < lx < XLIM[1] - LBL_W):
                        continue
                    if not (YLIM[0] + LBL_H < ly < YLIM[1] - LBL_H):
                        continue
                    if (_boxes_clear(lx, ly, boxes)
                            and _leader_clear(c.x, c.y, lx, ly, boxes)
                            and not _crosses((c.x, c.y), (lx, ly), segs)):
                        pick = (lx, ly)
                        break
                if pick:
                    break
            if pick is None:
                pick = (c.x, c.y + _RADII[-1])
            lx, ly = pick
            boxes.append((lx, ly))
            segs.append(((c.x, c.y), (lx, ly)))

            # Stop the leader at the label's edge so it never runs under the text.
            dx, dy = lx - c.x, ly - c.y
            nrm = (dx * dx + dy * dy) ** 0.5 or 1.0
            ux, uy = dx / nrm, dy / nrm
            t_stop = min(LBL_W / 2 / abs(ux) if ux else 1e9,
                         LBL_H / 2 / abs(uy) if uy else 1e9)
            ex, ey = lx - ux * t_stop, ly - uy * t_stop

            # White with a dark stroke: the labelled districts are the highest-value
            # ones, so they sit at the near-black end of the ramp and a dark leader
            # would disappear into them.
            ax.plot([c.x, ex], [c.y, ey], color="white", linewidth=1.4, zorder=6,
                    solid_capstyle="round",
                    path_effects=[pe.withStroke(linewidth=2.8,
                                                foreground="#2B2B29")])
            ax.plot(c.x, c.y, marker="o", markersize=4.2, color="white",
                    markeredgecolor="#2B2B29", markeredgewidth=0.9, zorder=7)
            ax.text(lx, ly,
                    f"{name_by_code.get(code, code)}\n{100*val/series.sum():.1f}%",
                    fontsize=9.0, fontweight="bold", ha="center", va="center",
                    color="white", zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.6,
                                                foreground="#2B2B29")])


    label_top_districts(ax_tot, s_tot, N_LABEL)

    cat_axes = []
    for i, (lbl, s, cat, col) in enumerate(panels[1:]):
        ax = fig.add_subplot(gs[i // 2, 2 + i % 2])
        draw(ax, s, lbl, col)
        cat_axes.append(ax)

    # ── Grouping bracket, total ──> the 2x2 block ─────────────────────────────────
    fig.canvas.draw()
    bb = ax_tot.get_position()
    boxes = [a.get_position() for a in cat_axes]
    x_br = min(b.x0 for b in boxes) - 0.017
    y_lo = min(b.y0 for b in boxes) + 0.03
    y_hi = max(b.y1 for b in boxes) - 0.03
    y_mid = (y_lo + y_hi) / 2

    fig.add_artist(Line2D([x_br, x_br], [y_lo, y_hi], transform=fig.transFigure,
                          color=FAINT, linewidth=1.8, solid_capstyle="round",
                          zorder=0))
    for yy in (y_lo, y_hi):
        fig.add_artist(Line2D([x_br, x_br + 0.012], [yy, yy],
                              transform=fig.transFigure, color=FAINT,
                              linewidth=1.8, solid_capstyle="round", zorder=0))
    # plain horizontal connector, meeting the vertical bracket in a clean T
    fig.add_artist(Line2D([bb.x1 + 0.006, x_br], [y_mid, y_mid],
                          transform=fig.transFigure, color=FAINT, linewidth=1.8,
                          solid_capstyle="round", zorder=0))

    # ── Shared colourbar ──────────────────────────────────────────────────────────
    cax = fig.add_axes([0.30, 0.072, 0.40, 0.019])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="min")
    cb.set_label("District biodiversity impact  (PDF·yr, log scale)",
                 fontsize=9.8, color=INK, labelpad=7)
    cb.ax.tick_params(labelsize=8.8, color=MUTED, labelcolor=MUTED)
    cb.outline.set_edgecolor(FAINT)
    cb.outline.set_linewidth(0.6)

    fig.legend(handles=[mpatches.Patch(facecolor=BASE, edgecolor=EDGE,
                                       label="no impact in this category")],
               loc="center left", bbox_to_anchor=(0.745, 0.081),
               frameon=False, fontsize=9, labelcolor=MUTED)

    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    for ext in ("png",):
        out = os.path.join(RESULTS_FIGURES_DIR, f"fig_3.{ext}")
        plt.savefig(out, dpi=600, facecolor="white", bbox_inches="tight")
        print(f"Saved -> {out}")
    plt.close()

    print(f"\nnational combined total: {combined.sum():.4e} PDF·yr")
    for lbl, s, cat, _ in panels[1:]:
        print(f"  {lbl:<22s} {s.sum():.4e}  ({100*s.sum()/combined.sum():7.4f}%)  "
              f"n={(s > 0).sum()}")
    print(f"\ntop {N_LABEL} combined districts:")
    for code, val in combined.sort_values(ascending=False).head(N_LABEL).items():
        print(f"  {name_by_code.get(code, code):<18s} {val:.4e}  "
              f"{100*val/combined.sum():5.2f}%")


def make_fig3c_hierarchy():
    """Figure 3c (exploratory) - two-level disaggregation tree.

    Integrated from fig3c_impact_hierarchy.py.
    """
    """
    fig3c_impact_hierarchy.py
    Standalone script — Figure 3c (exploratory): the combined regionalised
    biodiversity impact disaggregated in TWO steps — total, then realm
    (terrestrial vs freshwater), then the four spatial categories.

    Run from the python/ directory:
        conda run -n bw25-regional python fig3c_impact_hierarchy.py

    Output:
        results/figures/fig_3_c.png (+ .pdf)

    Everything else follows fig3_impact_disaggregation.py (version A): absolute
    PDF*yr, one shared log colour scale (1e-13..1e-5), grey = no impact,
    Chemba excluded from water, top districts labelled on the total panel.

    Layout: a three-column tree read left to right —
        a) Total  ─┬─ b) Terrestrial ── d) Land use
                   └─ c) Freshwater  ─┬─ e) Water consumption
                                      ├─ f) Eutrophication (N)
                                      └─ g) Eutrophication (P)
    Terrestrial = land use occupation (the only terrestrial regionalised
    category); Freshwater = water + eutrophication N + P. Middle- and
    leaf-column maps share one size; the hierarchy is carried by position and
    the connecting brackets, not by panel size.
    """
    import os, sys

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    gpd.options.io_engine = "fiona"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    from src.config import TZ_DISTRICTS_PATH, RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR

    SCENARIO = "baseline"
    WATER_EXCLUDE = {"TZ0107"}
    N_LABEL = 4

    INK   = "#2B2B29"
    MUTED = "#63635C"
    FAINT = "#9A9A92"
    BASE  = "#EAEAE4"
    EDGE  = "#FFFFFF"
    OUTL  = "#8A8A82"
    CMAP  = "magma_r"
    VMIN, VMAX = 1e-13, 1e-5

    # ── Data ──────────────────────────────────────────────────────────────────────
    gdf = gpd.read_file(TZ_DISTRICTS_PATH)
    outline = gdf.dissolve().boundary
    name_by_code = dict(zip(gdf.ADM2_PCODE, gdf.ADM2_EN))

    d = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "district_scores_all_scenarios.csv"))
    d = d[~((d.district_id.isin(WATER_EXCLUDE)) & (d.category == "Water consumption"))]
    d = d[d.scenario == SCENARIO]

    def cat_series(cat):
        return d[d.category == cat].set_index("district_id")["score"]

    combined = d.groupby("district_id")["score"].sum()
    terr = cat_series("Land use occupation")
    fw_cats = ["Water consumption", "FW eutrophication N", "FW eutrophication P"]
    fresh = (d[d.category.isin(fw_cats)]
             .groupby("district_id")["score"].sum())

    minx, miny, maxx, maxy = gdf.total_bounds
    mx, my = (maxx - minx) * 0.02, (maxy - miny) * 0.02
    XLIM, YLIM = (minx - mx, maxx + mx), (miny - my, maxy + my)

    # ── Figure ────────────────────────────────────────────────────────────────────
    # 15 rows: four 3-row leaf slots separated by 1-row gaps, so each map's
    # title sits in clear space instead of against the map above it.
    fig = plt.figure(figsize=(15.2, 10.6), dpi=200)
    fig.patch.set_facecolor("white")
    gs = GridSpec(15, 3, figure=fig, width_ratios=[2.35, 1.0, 1.0],
                  hspace=0.0, wspace=0.05,
                  left=0.020, right=0.980, top=0.960, bottom=0.100)

    norm = LogNorm(vmin=VMIN, vmax=VMAX, clip=True)
    cmap = matplotlib.colormaps.get_cmap(CMAP)


    def draw(ax, series, title, accent=INK, big=False):
        g = gdf.copy()
        g["v"] = g["ADM2_PCODE"].map(series)
        has = g["v"].notna() & (g["v"] > 0)
        g[~has].plot(ax=ax, facecolor=BASE, edgecolor=EDGE, linewidth=0.20, zorder=1)
        g[has].plot(ax=ax, column="v", cmap=cmap, norm=norm,
                    edgecolor=EDGE, linewidth=0.20, zorder=2)
        outline.plot(ax=ax, color=OUTL, linewidth=0.6, zorder=3)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=13.5 if big else 10.2, fontweight="bold",
                     color=INK, pad=9 if big else 4)
        if 0 < has.sum() <= 10:
            for geom in g.loc[has, "geometry"]:
                c = geom.representative_point()
                ax.plot(c.x, c.y, marker="o", markersize=8, markerfacecolor="none",
                        markeredgecolor=accent, markeredgewidth=1.3, zorder=6)
        return ax


    # panels — leaves at rows 0-3 / 4-7 / 8-11 / 12-15; terrestrial aligned
    # with its single child, freshwater centred on its three
    ax_tot   = fig.add_subplot(gs[:, 0])
    ax_terr  = fig.add_subplot(gs[0:3, 1])
    ax_fresh = fig.add_subplot(gs[8:11, 1])
    ax_land  = fig.add_subplot(gs[0:3, 2])
    ax_wat   = fig.add_subplot(gs[4:7, 2])
    ax_eun   = fig.add_subplot(gs[8:11, 2])
    ax_eup   = fig.add_subplot(gs[12:15, 2])

    draw(ax_tot, combined, "a) Total", big=True)
    draw(ax_terr, terr, "b) Terrestrial")
    draw(ax_fresh, fresh, "c) Freshwater", accent="#4E7291")
    draw(ax_land, terr, "d) Land use")
    draw(ax_wat, cat_series("Water consumption"), "e) Water consumption",
         accent="#4E7291")
    draw(ax_eun, cat_series("FW eutrophication N"), "f) Eutrophication (N)",
         accent="#C08A4E")
    draw(ax_eup, cat_series("FW eutrophication P"), "g) Eutrophication (P)",
         accent="#8E6086")

    # ── Top-district labels on the total (same algorithm as fig3) ────────────────
    LBL_W, LBL_H = 2.2, 1.35
    _ANGLES = [0, -30, 30, -60, 60, -90, 90, -125, 125, 180, -155, 155]
    _RADII = (3.4, 5.0, 6.8)


    def _boxes_clear(lx, ly, boxes):
        return not any(abs(lx - px) < LBL_W and abs(ly - py) < LBL_H
                       for px, py in boxes)


    def _leader_clear(cx, cy, lx, ly, boxes):
        for t in np.linspace(0.0, 1.0, 16):
            sx, sy = cx + t * (lx - cx), cy + t * (ly - cy)
            if any(abs(sx - px) < LBL_W * 0.62 and abs(sy - py) < LBL_H * 0.62
                   for px, py in boxes):
                return False
        return True


    def _crosses(p1, p2, segs):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        for p3, p4 in segs:
            if (ccw(p1, p3, p4) != ccw(p2, p3, p4)
                    and ccw(p1, p2, p3) != ccw(p1, p2, p4)):
                return True
        return False


    def label_top_districts(ax, series, n):
        boxes, segs = [], []
        for code, val in series.sort_values(ascending=False).head(n).items():
            geom = gdf.loc[gdf.ADM2_PCODE == code, "geometry"]
            if geom.empty:
                continue
            c = geom.iloc[0].representative_point()
            pick = None
            for r in _RADII:
                for a in _ANGLES:
                    lx = c.x + r * np.cos(np.radians(a))
                    ly = c.y + r * np.sin(np.radians(a))
                    if not (XLIM[0] + LBL_W < lx < XLIM[1] - LBL_W):
                        continue
                    if not (YLIM[0] + LBL_H < ly < YLIM[1] - LBL_H):
                        continue
                    if (_boxes_clear(lx, ly, boxes)
                            and _leader_clear(c.x, c.y, lx, ly, boxes)
                            and not _crosses((c.x, c.y), (lx, ly), segs)):
                        pick = (lx, ly)
                        break
                if pick:
                    break
            if pick is None:
                pick = (c.x, c.y + _RADII[-1])
            lx, ly = pick
            boxes.append((lx, ly))
            segs.append(((c.x, c.y), (lx, ly)))
            dx, dy = lx - c.x, ly - c.y
            nrm = (dx * dx + dy * dy) ** 0.5 or 1.0
            ux, uy = dx / nrm, dy / nrm
            t_stop = min(LBL_W / 2 / abs(ux) if ux else 1e9,
                         LBL_H / 2 / abs(uy) if uy else 1e9)
            ex, ey = lx - ux * t_stop, ly - uy * t_stop
            ax.plot([c.x, ex], [c.y, ey], color="white", linewidth=1.4, zorder=6,
                    solid_capstyle="round",
                    path_effects=[pe.withStroke(linewidth=2.8, foreground=INK)])
            ax.plot(c.x, c.y, marker="o", markersize=4.2, color="white",
                    markeredgecolor=INK, markeredgewidth=0.9, zorder=7)
            ax.text(lx, ly,
                    f"{name_by_code.get(code, code)}\n{100*val/series.sum():.1f}%",
                    fontsize=9.0, fontweight="bold", ha="center", va="center",
                    color="white", zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground=INK)])


    label_top_districts(ax_tot, combined, N_LABEL)

    # ── Connectors (fig3's bracket vocabulary: plain lines, clean T-joins) ───────
    fig.canvas.draw()
    TICK = 0.010


    def vline(x, y0, y1):
        fig.add_artist(Line2D([x, x], [y0, y1], transform=fig.transFigure,
                              color=FAINT, linewidth=1.8,
                              solid_capstyle="round", zorder=0))


    def hline(x0, x1, y):
        fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure,
                              color=FAINT, linewidth=1.8,
                              solid_capstyle="round", zorder=0))


    b_tot   = ax_tot.get_position()
    b_terr  = ax_terr.get_position()
    b_fresh = ax_fresh.get_position()
    b_land  = ax_land.get_position()
    b_wat   = ax_wat.get_position()
    b_eun   = ax_eun.get_position()
    b_eup   = ax_eup.get_position()

    yc = lambda b: (b.y0 + b.y1) / 2

    # total -> {terrestrial, freshwater}: the parent connector arrives at the
    # exact midpoint of the child bracket, so every junction is a clean T
    x1 = b_terr.x0 - 0.016
    y_branch1 = (yc(b_terr) + yc(b_fresh)) / 2
    hline(b_tot.x1 - 0.012, x1, y_branch1)
    vline(x1, yc(b_fresh), yc(b_terr))
    hline(x1, x1 + TICK, yc(b_terr))
    hline(x1, x1 + TICK, yc(b_fresh))

    # terrestrial -> land use (straight across, same row)
    hline(b_terr.x1 + 0.004, b_land.x0 - 0.004, yc(b_terr))

    # freshwater -> {water, eutro N, eutro P}. yc(freshwater) == yc(eutro N) ==
    # the bracket midpoint by construction, so the incoming connector and the
    # middle tick form one continuous straight line through the vertical.
    x2 = b_wat.x0 - 0.014
    hline(b_fresh.x1 + 0.004, x2, yc(b_fresh))
    vline(x2, yc(b_eup), yc(b_wat))
    hline(x2, x2 + TICK, yc(b_wat))
    hline(x2, x2 + TICK, yc(b_eup))
    hline(x2, x2 + TICK, yc(b_eun))

    # ── Shared colourbar + legend ────────────────────────────────────────────────
    cax = fig.add_axes([0.30, 0.050, 0.40, 0.016])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="min")
    cb.set_label("District biodiversity impact  (PDF·yr, log scale)",
                 fontsize=9.8, color=INK, labelpad=7)
    cb.ax.tick_params(labelsize=8.8, color=MUTED, labelcolor=MUTED)
    cb.outline.set_edgecolor(FAINT)
    cb.outline.set_linewidth(0.6)

    fig.legend(handles=[mpatches.Patch(facecolor=BASE, edgecolor=EDGE,
                                       label="no impact in this category")],
               loc="center left", bbox_to_anchor=(0.745, 0.058),
               frameon=False, fontsize=9, labelcolor=MUTED)

    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    for ext in ("png",):
        out = os.path.join(RESULTS_FIGURES_DIR, f"fig_3_c.{ext}")
        plt.savefig(out, dpi=600, facecolor="white", bbox_inches="tight")
        print(f"Saved -> {out}")
    plt.close()

    tot = combined.sum()
    print(f"\nnational combined total: {tot:.4e} PDF·yr")
    print(f"  terrestrial (land use)  {terr.sum():.4e}  ({100*terr.sum()/tot:8.4f}%)")
    print(f"  freshwater (W+N+P)      {fresh.sum():.4e}  ({100*fresh.sum()/tot:8.4f}%)")


def make_fig4_dotplot_uncertainty():
    """Figure 4 - cross-scenario dot plot with Monte Carlo intervals.

    Integrated from dot_plot_uncertainty.py.
    """
    import os as _os
    from src.config import RESULTS_TABLES_DIR as _RT
    _mc = [f for f in _os.listdir(_RT) if f.startswith('mc_')]
    if not _mc:
        print('make_fig4_dotplot_uncertainty: no mc_* tables in '
              'results/tables - run 02_monte_carlo first. SKIPPED.')
        return
    """
    dot_plot_uncertainty.py
    Cross-category, cross-scenario LCIA dot plot (log x-axis) WITH Monte Carlo
    95% uncertainty intervals.

    Same layout and central values as dot_plot.py, plus a horizontal whisker on
    each dot spanning the MC 2.5th-97.5th percentile.

    Consistency note on water: the deterministic dot uses the Chemba (TZ0107)
    excluded water score, but the Monte Carlo runs on the full spatial water score
    (Chemba included). We therefore anchor each interval on the plotted (central)
    value and apply the MC RELATIVE spread — lo = central * p2.5/median,
    hi = central * p97.5/median — so the band is always consistent with the dot,
    and reflects the background flow-distribution uncertainty rather than the
    Chemba offset.

    Reads results/tables/{all_scenarios,district_scores_all_scenarios,
    mc_summary_all_scenarios}.csv. No Brightway needed.
    Run:
        conda run -n bw25-regional python dot_plot_uncertainty.py

    Output:
        results/figures/fig_4.png
    """
    import sys, os

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    from src.config import RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR

    # ── Load data ──────────────────────────────────────────────────────────────────
    df          = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "all_scenarios.csv"))
    district_df = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "district_scores_all_scenarios.csv"))
    display_df  = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "scenario_display.csv"))
    mc          = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "mc_summary_all_scenarios.csv"))

    _WATER_EXCLUDE = {"TZ0107"}
    district_df = district_df[
        ~((district_df["district_id"].isin(_WATER_EXCLUDE)) &
          (district_df["category"] == "Water consumption"))
    ].copy()

    scenario_display = dict(zip(display_df["scenario"], display_df["display"]))
    _water = (district_df[district_df["category"] == "Water consumption"]
              .groupby("scenario")["score"].sum().to_dict())

    # (display label, all_scenarios category, MC category, source)
    _CATS = [
        ("Freshwater ecotoxicity",        "FW ecotoxicity",       "FW ecotoxicity",       "df"),
        ("Land use occupation",           "Land use occupation",  "Land use occupation",  "df"),
        ("Freshwater eutrophication (P)", "FW eutrophication P",  "FW eutrophication P",  "df"),
        ("Freshwater eutrophication (N)", "FW eutrophication N",  "FW eutrophication N",  "df"),
        ("Climate change",                "Climate change rcp26", "Climate change rcp26", "df"),
        ("Water consumption",             None,                   "Water consumption",    "water"),
    ]

    # Categories assessed with sub-national (ADM2 district) characterisation
    # factors; these get an asterisk on the y-axis. Climate change and freshwater
    # ecotoxicity have no spatially explicit sub-national CFs — ecotoxicity uses the
    # W6 continental box, climate a global factor — so they are national-average.
    # Define the asterisk in the figure caption.
    REGIONALIZED = {
        "Land use occupation",
        "Water consumption",
        "Freshwater eutrophication (N)",
        "Freshwater eutrophication (P)",
    }

    scores, mc_cat = {}, {}
    for label, cat, mccat, src in _CATS:
        scores[label] = dict(_water) if src == "water" else \
            dict(zip(df[df["category"] == cat]["scenario"], df[df["category"] == cat]["score"]))
        mc_cat[label] = mccat

    # MC relative-spread lookup: (scenario, mc_category) -> (p2.5/median, p97.5/median)
    mc_ratio = {}
    for _, r in mc.iterrows():
        med = r["median"]
        if med and med > 0:
            mc_ratio[(r["scenario"], r["category"])] = (r["p2.5"] / med, r["p97.5"] / med)

    # Freshwater ecotoxicity: override with the NUMERICALLY STABLE subset produced
    # by mc_ecotox_stable.py. On the full inventory this category has CV ~81% with
    # ~10% of draws negative, driven by roughly 40 flows that sample negative in
    # more than 1% of draws — ecoinvent waste-treatment activities carrying negative
    # scaling factors under the cutoff convention, amplified by very high metal CFs.
    # Dropping them removes ~76% of the variance for ~1.25% of the deterministic
    # score, leaves no negative draws, and brings CV to ~36%. Ecotoxicity is still
    # the widest of the six categories either way.
    _stable_path = os.path.join(RESULTS_TABLES_DIR, "mc_ecotox_stable.csv")
    if os.path.exists(_stable_path):
        _stab = pd.read_csv(_stable_path)
        for _, r in _stab.iterrows():
            med = r["stable_median"]
            if med and med > 0:
                mc_ratio[(r["scenario"], "FW ecotoxicity")] = (
                    r["stable_p2_5"] / med, r["stable_p97_5"] / med)
        print(f"ecotoxicity intervals: stable subset, "
              f"threshold={_stab['neg_rate_threshold'].iloc[0]}, "
              f"CV={_stab['stable_cv_pct'].mean():.1f}% "
              f"(full-inventory CV was {_stab['full_cv_pct'].mean():.1f}%)")
    else:
        print("WARNING: mc_ecotox_stable.csv not found — ecotoxicity intervals "
              "fall back to the full inventory and will be open-ended")

    _cat_labels = sorted([c[0] for c in _CATS],
                         key=lambda l: scores[l].get("baseline", 0.0), reverse=True)

    _SCEN_ORDER = ["baseline", "high_yield", "extensification", "organic_expansion",
                   "manufacturing_expansion", "irrigation"]
    _COLORS = {"baseline": "#4D4D4D", "high_yield": "#1F77B4", "extensification": "#8C564B",
               "organic_expansion": "#2CA02C", "manufacturing_expansion": "#9467BD",
               "irrigation": "#FF7F0E"}
    scenarios = [s for s in _SCEN_ORDER if s in scenario_display]

    # ── Figure ────────────────────────────────────────────────────────────────────
    n_cat, n_scen = len(_cat_labels), len(scenarios)
    offsets = np.linspace(0.20, -0.20, n_scen)

    fig, ax = plt.subplots(figsize=(9.6, 0.98 * n_cat + 1.9), dpi=200)
    fig.patch.set_facecolor("white")

    # x-range accounts for the whisker extents, not just the dots
    _pts = []
    for lbl in _cat_labels:
        for s in scenarios:
            v = scores[lbl].get(s)
            if not v or v <= 0:
                continue
            _pts.append(v)
            lohi = mc_ratio.get((s, mc_cat[lbl]))
            if lohi:
                # Only positive endpoints can sit on a log axis. Freshwater
                # ecotoxicity has a negative 2.5th percentile in every scenario
                # (see the open-whisker handling below), so its lower bound is
                # excluded from the range calculation rather than crashing it.
                _pts.extend([p for p in (v * lohi[0], v * lohi[1]) if p > 0])
    x_lo, x_hi = min(_pts) / 2.2, max(_pts) * 2.2

    for i, lbl in enumerate(_cat_labels):
        y = n_cat - 1 - i
        ax.plot([x_lo, x_hi], [y, y], linestyle=(0, (1, 4)), linewidth=0.7,
                color="#C8C8C4", zorder=1)

        for j, scen in enumerate(scenarios):
            val = scores[lbl].get(scen)
            if val is None or val <= 0:
                continue
            yj = y + offsets[j]
            color = _COLORS.get(scen, f"C{j}")

            lohi = mc_ratio.get((scen, mc_cat[lbl]))
            if lohi:
                lo, hi = val * lohi[0], val * lohi[1]
                cap = 0.05
                # A non-positive lower bound cannot be drawn on a log axis. That
                # happens for freshwater ecotoxicity in every scenario: ~10% of
                # draws are negative, because ecoinvent carries ~4,600 normal
                # distributions narrow enough to sample below zero, and the
                # category is dominated by a few very high-CF metal flows. Draw
                # the whisker open to the left edge instead of silently dropping
                # it, and omit the lower cap so the reader sees it is unbounded.
                open_lo = lo <= 0
                lo_draw = x_lo if open_lo else lo
                ax.plot([lo_draw, hi], [yj, yj], color=color, linewidth=1.3,
                        alpha=0.7, zorder=2, solid_capstyle="round")
                ax.plot([hi, hi], [yj - cap, yj + cap], color=color,
                        linewidth=1.1, alpha=0.7, zorder=2)
                if not open_lo:
                    ax.plot([lo, lo], [yj - cap, yj + cap], color=color,
                            linewidth=1.1, alpha=0.7, zorder=2)
            ax.plot(val, yj, marker="o", markersize=6.6, color=color,
                    markeredgecolor="white", markeredgewidth=0.9,
                    linestyle="none", zorder=3)

    for i in range(n_cat - 1):
        ax.axhline(n_cat - 1.5 - i, linestyle=":", linewidth=1.0,
                   color="#A9A9A4", zorder=1.5)

    ax.set_xscale("log")
    ax.grid(axis="x", which="major", linestyle="-", linewidth=0.5,
            color="#E6E6E2", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.7, n_cat - 1 + 0.7)
    ax.set_yticks([n_cat - 1 - i for i in range(n_cat)])
    # Asterisk marks the regionalised categories. _cat_labels itself is left alone:
    # it is used as a dict key into `scores` and `mc_cat` further up.
    ax.set_yticklabels([f"{l} *" if l in REGIONALIZED else l for l in _cat_labels],
                       fontsize=9.5)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Biodiversity damage (PDF·yr, log scale)  ·  whiskers = 95% Monte Carlo "
                  "interval (N = 1,000)",
                  fontsize=9.5, labelpad=8)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#8A8A85")
    ax.spines["bottom"].set_linewidth(0.8)

    handles = [mlines.Line2D([], [], color=_COLORS.get(s, "grey"), marker="o",
               markersize=7.5, markeredgecolor="white", markeredgewidth=0.9,
               linestyle="none", label=scenario_display.get(s, s)) for s in scenarios]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=n_scen, frameon=False, fontsize=9,
              handletextpad=0.35, columnspacing=1.4)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_FIGURES_DIR, "fig_4.png")
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out_path}")


def make_fig5_delta_maps():
    """Figure 5 - per-scenario delta maps vs baseline.

    Integrated from fig4_scenario_delta_maps.py.
    """
    """
    fig4_scenario_delta_maps.py
    Standalone script — Figure 4: district-level change in combined regionalised
    biodiversity impact, for every scenario against the baseline.

    Run from the python/ directory:
        conda run -n bw25-regional python fig4_scenario_delta_maps.py

    Output:
        results/figures/fig_5.png (+ .pdf)

    Reads results/tables/district_scores_all_scenarios.csv. No Brightway needed.

    Design notes
    ------------
    * Six panels, one per scenario, INCLUDING baseline. The baseline panel is zero
      everywhere by construction; keeping it makes the reference explicit and gives
      the grid a complete set.
    * ONE shared colour scale for all six panels, so they are directly comparable.
      Per-panel colourbars would let each scenario rescale itself and destroy the
      comparison the figure exists to make.
    * Combined impact is the plain ABSOLUTE sum of the four regionalised
      categories, in PDF*yr. No normalisation. An earlier version normalised each
      category to its baseline national total before summing, which silently gave
      water — 0.001% of baseline damage — the same weight as land use, which is
      99.99% of it. Under that weighting irrigation appeared to be the largest
      scenario by far (x50) purely because its water impact rose 198x from a
      negligible base; in absolute terms irrigation changes the regionalised total
      by 0.2%, and extensification is the only scenario that materially moves it
      (x3.90, from roughly quadrupling land occupation).
    * The scale is logarithmic, spanning 1e-12 to 1e-5 PDF*yr. Positive district
      deltas span about fifteen orders of magnitude, and extensification's largest
      district is four orders above the largest district of any other scenario. On
      a linear ramp every scenario except extensification would be invisible.
    * Four districts show tiny NEGATIVE deltas (3 in high_yield, 1 in irrigation),
      the largest being -3.2e-15 against a maximum of +1.8e-05. These are
      floating-point noise, not modelled decreases; they fall below the colour
      scale's floor and are drawn as "negligible change".
    * Per-panel statistics are printed to stdout, not drawn on the figure.
    * Chemba (TZ0107) is excluded from water, as in every other figure.
    """
    import os, sys

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    gpd.options.io_engine = "fiona"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches

    from src.config import TZ_DISTRICTS_PATH, RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR

    WATER_EXCLUDE = {"TZ0107"}
    BASELINE = "baseline"

    INK   = "#2B2B29"
    MUTED = "#63635C"
    FAINT = "#9A9A92"
    BASE  = "#EAEAE4"     # negligible / no change
    EDGE  = "#FFFFFF"
    OUTL  = "#8A8A82"
    CMAP  = "YlOrRd"

    VMIN, VMAX = 1e-12, 1e-5   # PDF*yr of increase

    ORDER = ["baseline", "high_yield", "extensification",
             "organic_expansion", "manufacturing_expansion", "irrigation"]
    SCEN_COLOR = {
        "baseline":                "#4D4D4D",
        "high_yield":              "#1F77B4",
        "extensification":         "#8C564B",
        "organic_expansion":       "#2CA02C",
        "manufacturing_expansion": "#9467BD",
        "irrigation":              "#FF7F0E",
    }

    # ── Data ──────────────────────────────────────────────────────────────────────
    gdf = gpd.read_file(TZ_DISTRICTS_PATH)
    outline = gdf.dissolve().boundary

    d = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "district_scores_all_scenarios.csv"))
    d = d[~((d.district_id.isin(WATER_EXCLUDE)) & (d.category == "Water consumption"))]

    disp = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "scenario_display.csv"))
    display = dict(zip(disp["scenario"], disp["display"]))

    # ABSOLUTE sum of the four categories, in PDF*yr. No normalisation: all four
    # are already PDF*yr, and normalising each to its own national total silently
    # gave water (0.001% of baseline damage) the same weight as land use (99.99%).
    piv = d.groupby(["scenario", "district_id"])["score"].sum().unstack(0).fillna(0.0)
    base_col = piv[BASELINE]
    NATIONAL_BASE = base_col.sum()

    delta = {s: piv[s] - base_col for s in piv.columns}
    fold = {s: piv[s].sum() / NATIONAL_BASE for s in piv.columns}

    minx, miny, maxx, maxy = gdf.total_bounds
    mx, my = (maxx - minx) * 0.02, (maxy - miny) * 0.02
    XLIM, YLIM = (minx - mx, maxx + mx), (miny - my, maxy + my)

    # ── Figure ────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15.0, 9.6), dpi=200)
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 3, figure=fig, hspace=0.12, wspace=0.02,
                  left=0.018, right=0.982, top=0.945, bottom=0.155)

    norm = LogNorm(vmin=VMIN, vmax=VMAX, clip=True)
    cmap = matplotlib.colormaps.get_cmap(CMAP)

    for i, scen in enumerate(ORDER):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        dv = delta[scen]

        g = gdf.copy()
        g["v"] = g["ADM2_PCODE"].map(dv)
        changed = g["v"].notna() & (g["v"] > VMIN)

        g[~changed].plot(ax=ax, facecolor=BASE, edgecolor=EDGE, linewidth=0.22,
                         zorder=1)
        if changed.any():
            g[changed].plot(ax=ax, column="v", cmap=cmap, norm=norm,
                            edgecolor=EDGE, linewidth=0.22, zorder=2)
        outline.plot(ax=ax, color=OUTL, linewidth=0.7, zorder=3)

        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.set_axis_off()

        # Scenario name in a box outlined in that scenario's colour, matching the
        # palette used by every other figure in the manuscript.
        col = SCEN_COLOR.get(scen, INK)
        ax.text(0.5, 1.035, display.get(scen, scen), transform=ax.transAxes,
                fontsize=12.4, fontweight="bold", color=INK,
                ha="center", va="bottom", zorder=7, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.28",
                          facecolor="white", edgecolor=col, linewidth=1.9))

    # Per-panel statistics (national fold, districts changed, largest delta) are
    # printed to stdout rather than drawn on the figure — they belong in the caption
    # or the text, not as grey clutter under every map.

    # ── Shared colourbar ──────────────────────────────────────────────────────────
    cax = fig.add_axes([0.28, 0.082, 0.44, 0.018])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Increase in combined biodiversity impact vs baseline  "
                 "(PDF·yr, log scale)",
                 fontsize=9.8, color=INK, labelpad=7)
    cb.ax.tick_params(labelsize=8.8, color=MUTED, labelcolor=MUTED)
    cb.outline.set_edgecolor(FAINT)
    cb.outline.set_linewidth(0.6)

    fig.legend(handles=[mpatches.Patch(facecolor=BASE, edgecolor=EDGE,
                                       label="no or negligible change")],
               loc="center left", bbox_to_anchor=(0.745, 0.091),
               frameon=False, fontsize=9, labelcolor=MUTED)

    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    for ext in ("png",):
        out = os.path.join(RESULTS_FIGURES_DIR, f"fig_5.{ext}")
        plt.savefig(out, dpi=600, facecolor="white", bbox_inches="tight")
        print(f"Saved -> {out}")
    plt.close()

    print(f"\nbaseline national combined impact: {NATIONAL_BASE:.4e} PDF·yr")
    print(f"{'scenario':<26s}{'fold':>7s}{'n changed':>11s}{'max':>10s}{'min':>12s}")
    for s in ORDER:
        dv = delta[s]
        print(f"{s:<26s}{fold[s]:>7.3f}{int((dv > VMIN).sum()):>11d}"
              f"{dv.max():>10.2e}{dv.min():>12.2e}")


def make_fig6_dls_spider():
    """Figure 6 - DLS spider, all scenarios, baseline inset.

    Integrated from dls_spider_combined.py.
    """
    """
    dls_spider_combined.py
    Standalone script — single radar (spider) plot with ALL scenarios overlaid,
    showing Decent Living Standards coverage across the five DLS indicators.

    DLS coverage is bounded 0-1, which is what makes a radar appropriate here
    (unlike the biodiversity impacts, which span ~7 orders of magnitude and are
    better served by the log dot plot in dot_plot.py).

    The radial axis runs 0-1, where 1.0 = the decent-living threshold is fully
    met. Plotting against the full threshold rather than the data maximum keeps
    the gap to decent living visible, which is itself a central finding.

    Design notes (manuscript revision):
    - The baseline polygon is buried under the scenario overlays, so a zoomed
      inset (upper-left, radial axis 0-0.25) shows it alone.
    - Each scenario's highest indicator carries a small value label at its
      vertex; collisions on the shared Farm-income axis are offset tangentially.
    - No figure title: the manuscript caption carries it, as with figures 1-5.

    Reads pre-computed CSVs from results/tables/. No Brightway needed.
    Run from the python/ directory:
        conda run -n bw25-regional python dls_spider_combined.py

    Output:
        results/figures/fig_6.png
    """
    import sys, os

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects
    import matplotlib.pyplot as plt

    from src.config import RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR

    # ── Load data ──────────────────────────────────────────────────────────────────
    dls_df     = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "dls_coverage_all_scenarios.csv"))
    display_df = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "scenario_display.csv"))

    scenario_display = dict(zip(display_df["scenario"], display_df["display"]))

    # Indicator order around the circle (grouped: nutrition, material, economic)
    _IND_ORDER = [
        "Nutrition (kcal)",
        "Nutrition (protein)",
        "Clothing",
        "Farm income (living-income)",
        "Employment (rural labour)",
    ]
    # Shorter labels for the plot
    _IND_LABELS = {
        "Nutrition (kcal)":            "Nutrition\n(kcal)",
        "Nutrition (protein)":         "Nutrition\n(protein)",
        "Clothing":                    "Clothing",
        "Farm income (living-income)": "Farm income\n(living-income)",
        "Employment (rural labour)":   "Employment\n(rural labour)",
    }

    _SCEN_ORDER = ["baseline", "high_yield", "extensification", "organic_expansion",
                   "manufacturing_expansion", "irrigation"]
    _COLORS = {
        "baseline":                "#4D4D4D",
        "high_yield":              "#1F77B4",
        "extensification":         "#8C564B",
        "organic_expansion":       "#2CA02C",
        "manufacturing_expansion": "#9467BD",
        "irrigation":              "#FF7F0E",
    }

    cov = dls_df.pivot_table(index="indicator", columns="scenario",
                             values="coverage", aggfunc="first")

    indicators = [i for i in _IND_ORDER if i in cov.index]
    scenarios  = [s for s in _SCEN_ORDER if s in cov.columns]

    # ── Geometry ──────────────────────────────────────────────────────────────────
    n = len(indicators)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]          # close the polygon

    GRID = "#C2C2BB"                             # darker than before, was #E4E4E0

    # Square canvas: the exported PNG then has a 1:1-friendly aspect, so a
    # mis-sized frame in Word cannot squash the circles into ellipses as badly,
    # and the inset gets guaranteed headroom.
    fig = plt.figure(figsize=(7.8, 7.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.13, 0.10, 0.74, 0.74], projection="polar")

    ax.set_theta_offset(np.pi / 2)               # first axis at top
    ax.set_theta_direction(-1)                   # clockwise

    for scen in scenarios:
        vals = [float(cov.loc[ind, scen]) for ind in indicators]
        vals_closed = vals + vals[:1]
        is_base = scen == "baseline"
        color = _COLORS.get(scen, "grey")

        # Soft translucent fill + thin outline, no markers
        ax.fill(angles_closed, vals_closed,
                color=color, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(angles_closed, vals_closed,
                color=color,
                linewidth=1.3 if is_base else 1.5,
                linestyle=(0, (4, 2)) if is_base else "-",
                solid_joinstyle="round",
                label="Baseline (inset)" if is_base else scenario_display.get(scen, scen),
                zorder=3)

    # ── Value label on each scenario's highest indicator ──────────────────────────
    # Several scenarios peak on the same Farm-income axis, so labels there are
    # fanned tangentially/radially to keep them apart.  (dr, dth) offsets in
    # (radial units, radians).
    _OFFSETS = {
        "high_yield":              (0.055, 0.00),
        "irrigation":              (0.055, 0.00),
        "manufacturing_expansion": (0.055, 0.00),
        "extensification":         (0.055, -0.16),
        "organic_expansion":       (0.055, 0.16),
    }
    for scen in scenarios:
        if scen == "baseline":
            continue
        vals = [float(cov.loc[ind, scen]) for ind in indicators]
        k = int(np.argmax(vals))
        dr, dth = _OFFSETS.get(scen, (0.055, 0.0))
        ax.text(angles[k] + dth, vals[k] + dr, f"{vals[k]:.2f}",
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=_COLORS[scen], zorder=6,
                path_effects=[matplotlib.patheffects.withStroke(
                    linewidth=2.4, foreground="white")])

    # ── Axes / chrome ─────────────────────────────────────────────────────────────
    ax.set_xticks(angles)
    ax.set_xticklabels([_IND_LABELS.get(i, i) for i in indicators], fontsize=9,
                       color="#3A3A38")
    ax.tick_params(axis="x", pad=16)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.5,
                       color="#6E6E68")
    ax.set_rlabel_position(180 / n)

    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color(GRID)
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_facecolor("white")

    # ── Baseline inset (upper-left dead space, radial axis zoomed to 0-0.25) ──────
    # Positioned high enough to clear the main plot's Employment axis label,
    # but with its title safely inside the canvas.
    axi = fig.add_axes([0.005, 0.715, 0.235, 0.235], projection="polar")
    axi.set_theta_offset(np.pi / 2)
    axi.set_theta_direction(-1)
    bvals = [float(cov.loc[ind, "baseline"]) for ind in indicators]
    bclosed = bvals + bvals[:1]
    axi.fill(angles_closed, bclosed, color=_COLORS["baseline"], alpha=0.18,
             linewidth=0, zorder=2)
    axi.plot(angles_closed, bclosed, color=_COLORS["baseline"], linewidth=1.4,
             linestyle=(0, (4, 2)), zorder=3)
    axi.set_ylim(0, 0.25)
    axi.set_yticks([0.1, 0.2])
    axi.set_yticklabels(["0.1", "0.2"], fontsize=6, color="#6E6E68")
    axi.set_rlabel_position(180 / n)
    axi.set_xticks(angles)
    axi.set_xticklabels(["Kcal", "Prot.", "Cloth.", "Income", "Empl."],
                        fontsize=6.5, color="#3A3A38")
    axi.tick_params(axis="x", pad=1)
    axi.grid(color=GRID, linewidth=0.6)
    axi.spines["polar"].set_color("#9A9A95")
    axi.spines["polar"].set_linewidth(0.8)
    axi.set_facecolor("#FCFCFA")
    # label the baseline's own peak (farm income) inside the inset
    kb = int(np.argmax(bvals))
    axi.text(angles[kb], bvals[kb] + 0.045, f"{bvals[kb]:.2f}",
             ha="center", va="center", fontsize=7, fontweight="bold",
             color=_COLORS["baseline"], zorder=6,
             path_effects=[matplotlib.patheffects.withStroke(
                 linewidth=2.0, foreground="#FCFCFA")])
    axi.set_title("Baseline (axis 0–0.25)", fontsize=8, color="#3A3A38",
                  pad=6)

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06),
                    ncol=3, frameon=False, fontsize=9,
                    handletextpad=0.7, columnspacing=2.0, handlelength=2.2)
    for txt in leg.get_texts():
        txt.set_color("#3A3A38")

    out_path = os.path.join(RESULTS_FIGURES_DIR, "fig_6.png")
    plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.close()
    print(f"Saved -> {out_path}")

    print("\nDLS coverage (fraction of threshold met):")
    hdr = "  " + " " * 30 + "".join(f"{scenario_display.get(s, s)[:13]:>15s}" for s in scenarios)
    print(hdr)
    for ind in indicators:
        row = "".join(f"{float(cov.loc[ind, s]):>15.3f}" for s in scenarios)
        print(f"  {ind:30s}{row}")
    means = "".join(f"{float(cov[s].mean()):>15.3f}" for s in scenarios)
    print(f"  {'MEAN':30s}{means}")


def make_fig7_tradeoff():
    """Figure 7 - well-being vs biodiversity trade-off + composition key.

    Integrated from fig5_tradeoff.py.
    """
    """
    fig5_tradeoff.py
    Standalone script — Figure 5: the well-being / biodiversity trade-off scatter
    with a composition key. Replaces scenario_tradeoff.png as the manuscript's
    trade-off figure.

    Run from the python/ directory:
        conda run -n bw25-regional python fig5_tradeoff.py

    Output:
        results/figures/fig_7.png (+ .pdf)

    Reads results/tables/. No Brightway needed.

    Design
    ------
    Panel (a): the established trade-off axes — x = mean of per-indicator DLS
    changes (%), y = geometric mean of per-category impact fold-changes (%), water
    corrected for the Chemba outlier as everywhere else. Only the baseline dot is
    labelled; panel (b) identifies the rest.

    Panel (b): one row per scenario (baseline omitted — every bar would be zero),
    ordered top-to-bottom by panel (a)'s vertical position so the eye maps dot to
    row without search. Each row carries a composition glyph: five green bars are
    the DLS gains in percentage points; six red bars are the impact fold-changes,
    drawn at log height so water's x198 under irrigation does not flatten the
    rest. The key doubles as the legend, so panel (a) needs no labels.

    No figure title: the caption carries it, as with figures 1-4.
    """
    import os, sys

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.config import RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR

    INK, MUTED, LINE = "#2B2B29", "#63635C", "#DCDCD4"
    GREEN, RED = "#2E7D4F", "#B03A2E"
    SCOL = {"baseline": "#4D4D4D", "high_yield": "#1F77B4",
            "extensification": "#8C564B", "organic_expansion": "#2CA02C",
            "manufacturing_expansion": "#9467BD", "irrigation": "#FF7F0E"}
    SCEN = ["baseline", "high_yield", "extensification", "organic_expansion",
            "manufacturing_expansion", "irrigation"]
    CATS = ["Land use occupation", "FW ecotoxicity", "FW eutrophication P",
            "FW eutrophication N", "Climate change rcp26", "Water consumption"]
    CAT_AB = ["LU", "ET", "P", "N", "CC", "W"]
    IND = ["Farm income (living-income)", "Clothing", "Employment (rural labour)",
           "Nutrition (kcal)", "Nutrition (protein)"]
    IND_AB = ["In", "Cl", "Em", "Kc", "Pr"]

    # ── Data ─────────────────────────────────────────────────────────────────────
    imp = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "all_scenarios.csv"))
    dis = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "district_scores_all_scenarios.csv"))
    dls = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "dls_coverage_all_scenarios.csv"))
    disp = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "scenario_display.csv"))
    display = dict(zip(disp["scenario"], disp["display"]))

    piv = imp.pivot_table(index="category", columns="scenario", values="score",
                          aggfunc="first")
    piv.loc["Water consumption"] = (
        dis[(dis.category == "Water consumption") & (dis.district_id != "TZ0107")]
        .groupby("scenario")["score"].sum())

    dls["indicator"] = dls["indicator"].str.replace("\n", " ", regex=False)
    dpv = dls.pivot_table(index="indicator", columns="scenario", values="coverage",
                          aggfunc="first")

    folds = {s: [piv.loc[c, s] / piv.loc[c, "baseline"] for c in CATS] for s in SCEN}
    gains = {s: [(dpv.loc[i, s] - dpv.loc[i, "baseline"]) * 100 for i in IND]
             for s in SCEN}

    # axes values, matching scenario_tradeoff.py conventions
    wb = {s: 100 * np.mean([(dpv.loc[i, s] - dpv.loc[i, "baseline"])
                            / dpv.loc[i, "baseline"] for i in IND]) for s in SCEN}
    bio = {s: 100 * (np.exp(np.mean(np.log(folds[s]))) - 1) for s in SCEN}

    # Panel (b) units: ONE unit and ONE linear scale for both bar families —
    # % change vs baseline. (Percentage points only exist for DLS coverage,
    # which is a bounded percentage; impacts have no such scale, so % change is
    # the only unit both sides can share.) The scale is set by the bulk of the
    # data, NOT by irrigation's water outlier (+19,700% off a near-zero
    # rain-fed baseline) — that bar is capped with a break marker instead.
    gpct = {s: [(dpv.loc[i, s] / dpv.loc[i, "baseline"] - 1) * 100 for i in IND]
            for s in SCEN}
    rpct = {s: [(f - 1) * 100 for f in folds[s]] for s in SCEN}
    _bulk = max(max(gpct[s]) for s in SCEN)          # ≈ +291%, ex-outlier
    MAXPCT = _bulk * 1.10
    GRIDS = [(100.0, "+100%"), (200.0, "+200%"), (300.0, "+300%")]
    CAP = 1.04                                 # bar height at which we break

    # ── Figure ───────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11.6, 5.6), dpi=200)
    fig.patch.set_facecolor("white")

    # (a) scatter — slightly narrower, ceding room to the composition key
    ax = fig.add_axes([0.065, 0.125, 0.455, 0.845])
    xmax = max(wb.values()) * 1.10
    ymax = max(bio.values()) * 1.12
    ax.set_xlim(-xmax * 0.06, xmax)
    ax.set_ylim(-ymax * 0.07, ymax)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    for s in SCEN:
        ax.scatter(wb[s], bio[s], s=170, color=SCOL[s], edgecolor="white",
                   lw=1.8, zorder=5)
    # scenario labels on the scatter itself; the key column carries only dots.
    # Per-point offsets keep neighbours (organic/mfg cluster) apart.
    LBL_OFF = {"baseline": (10, 9), "high_yield": (-8, 12),
               "extensification": (-14, -20), "organic_expansion": (-4, 16),
               "manufacturing_expansion": (8, -18), "irrigation": (10, 3)}
    LBL_HA = {"high_yield": "right", "extensification": "right"}
    LBL_TXT = {"manufacturing_expansion": "Manufacturing Expansion"}
    for s in SCEN:
        ax.annotate(LBL_TXT.get(s, display.get(s, s)), (wb[s], bio[s]),
                    LBL_OFF[s], textcoords="offset points", fontsize=9,
                    fontweight="bold", color=SCOL[s], ha=LBL_HA.get(s, "left"))
    ax.set_xlabel("Well-being: mean change across five DLS indicators (%)",
                  fontsize=10.5)
    ax.set_ylabel("Biodiversity impact: geometric-mean change\nacross six "
                  "categories (%)", fontsize=10.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=9.5)
    # panel tag well above the axes, level with panel (b)'s tag

    # (b) composition key — baseline omitted, rows ordered by panel (a) height.
    # Rows are packed tightly and the scenario name sits vertically centred on
    # its glyph, so the dot/name column no longer floats in blank space.
    rows = sorted([s for s in SCEN if s != "baseline"], key=lambda s: -bio[s])
    KX, KW = 0.565, 0.415
    GX, GWg = KX + 0.032, KW - 0.032          # dot-only key: glyphs take the width
    fig.text(0.055, 0.985, "(a)", fontsize=11, fontweight="bold", va="bottom")
    fig.text(KX, 0.985, "(b)", fontsize=11, fontweight="bold", va="bottom")
    # group labels are placed BELOW the stack after layout, centred on the bar
    # groups and vertically level with panel (a)'s x-axis label (see the end of
    # this block). Glyph x-range is -0.7..12.3; green bars centre on 2.2, red
    # bars on 8.9.
    _frac = lambda gx: GX + GWg * (gx + 0.7) / 13.0

    # Rows aligned to panel (a): top of the first row meets the top of (a)'s
    # axes (0.970), the bottom row's baseline meets (a)'s bottom spine (0.125),
    # and the slack goes into wider gaps between rows.
    ROW_H, ROW_GAP, TOP = 0.1434, 0.032, 0.970
    for r, s in enumerate(rows):
        y0 = TOP - r * (ROW_H + ROW_GAP) - ROW_H
        # dot + name, centred on the glyph row. A Circle in figure fractions
        # renders as an ellipse on a non-square canvas, so compensate the height
        # by the figure aspect ratio.
        fw, fh = fig.get_size_inches()
        fig.patches.append(matplotlib.patches.Ellipse(
            (KX + 0.010, y0 + ROW_H * 0.42), width=0.0144,
            height=0.0144 * fw / fh, transform=fig.transFigure, color=SCOL[s]))
        # glyph — one linear % change scale for both halves, shared gridlines,
        # labels on the top row only
        g = fig.add_axes([GX, y0, GWg, ROW_H])
        g.set_xlim(-0.7, 12.3)
        g.set_ylim(0, 1.10)
        g.axis("off")
        for gv, glab in GRIDS:
            gy = gv / MAXPCT
            g.plot([-0.7, 12.3], [gy, gy], color="#C6C6BE", lw=0.7, zorder=1)
            if r == 0:
                g.text(12.4, gy, glab, ha="left", va="center", fontsize=6.2,
                       color=MUTED, clip_on=False)
        for j, v in enumerate(gpct[s]):
            g.bar(j, max(0, v) / MAXPCT, width=0.72, color=GREEN, alpha=0.92,
                  zorder=3)
        g.axvline(5.05, color=LINE, lw=0.9)
        for j, v in enumerate(rpct[s]):
            h = max(0, v) / MAXPCT
            xw = 5.6 + j + 0.5
            if h <= CAP:
                g.bar(xw, h, width=0.72, color=RED, alpha=0.92, zorder=3)
            else:
                # off-scale outlier: capped bar with a break marker
                g.bar(xw, CAP, width=0.72, color=RED, alpha=0.92, zorder=3)
                for dy in (0.0, 0.045):
                    g.plot([xw - 0.5, xw + 0.5], [0.80 + dy, 0.84 + dy],
                           color="white", lw=2.0, zorder=4, clip_on=False)
        # black x- and y-axis spines, same weight as panel (a)'s. Drawn as
        # unclipped lines: axhline at y=0 sits on the clip boundary and loses
        # half its width, which made the baselines render grey on some rows.
        g.plot([-0.7, 12.3], [0, 0], color="black", lw=0.8, zorder=4,
               clip_on=False, solid_capstyle="butt")
        g.plot([-0.7, -0.7], [0, 1.0], color="black", lw=0.8, zorder=4,
               clip_on=False, solid_capstyle="butt")
        # bar letters under the bottom row only, where nothing follows them.
        # Font size and colour match panel (a)'s tick labels, with short black
        # tick marks tying each label to its bar column.
        if r == len(rows) - 1:
            for j, ab in enumerate(IND_AB):
                g.plot([j, j], [0, -0.055], color="black", lw=0.8,
                       clip_on=False, zorder=4)
                g.text(j, -0.10, ab, ha="center", va="top", fontsize=9.5,
                       color="black", clip_on=False)
            for j, ab in enumerate(CAT_AB):
                xw = 5.6 + j + 0.5
                g.plot([xw, xw], [0, -0.055], color="black", lw=0.8,
                       clip_on=False, zorder=4)
                g.text(xw, -0.10, ab, ha="center", va="top", fontsize=9.5,
                       color="black", clip_on=False)

    # group labels below the stack, on exactly the same line as panel (a)'s
    # x-axis label (computed from the rendered label position, not eyeballed)
    fig.canvas.draw()
    _ext = ax.xaxis.label.get_window_extent()
    _ylab = fig.transFigure.inverted().transform((0, (_ext.y0 + _ext.y1) / 2))[1]
    fig.text(_frac(2.2), _ylab, "DLS (% change)", fontsize=10.5, color="black",
             ha="center", va="center")
    fig.text(_frac(8.9), _ylab, "LCIA (% change)", fontsize=10.5, color="black",
             ha="center", va="center")

    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    for ext in ("png",):
        out = os.path.join(RESULTS_FIGURES_DIR, f"fig_7.{ext}")
        fig.savefig(out, dpi=600, facecolor="white", bbox_inches="tight")
        print(f"Saved -> {out}")
    plt.close(fig)

    print(f"\n{'scenario':<26s}{'wb %':>9s}{'bio %':>9s}   folds / gains(pp)")
    for s in SCEN:
        fstr = " ".join(f"{f:5.2f}" for f in folds[s])
        gstr = " ".join(f"{v:4.1f}" for v in gains[s])
        print(f"{s:<26s}{wb[s]:>9.1f}{bio[s]:>9.1f}   [{fstr}] [{gstr}]")


def make_si_chemba_map():
    """SI - location of the Chemba water-CF outlier district.

    Integrated from si_chemba_map.py.
    """
    """
    si_chemba_map.py
    Standalone script — SI figure: location of Chemba District (TZ0107), the
    water-consumption outlier excluded from the district water totals in every
    main-text figure.

    Chemba's PCR-GLOBWB basin characterisation factor is several orders of
    magnitude above neighbouring districts, so in the irrigation scenario it alone
    carries ~99.9% of the national water-consumption score. This map shows where it
    sits relative to the cotton-producing districts.

    Run from the python/ directory:
        conda run -n bw25-regional python si_chemba_map.py

    Output:
        results/figures/fig_S4.png (+ .pdf)
    """
    import os, sys

    import geopandas as gpd
    gpd.options.io_engine = "fiona"
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches

    from src.config import TZ_DISTRICTS_PATH, RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR

    CHEMBA = "TZ0107"

    INK      = "#2B2B29"
    MUTED    = "#63635C"
    BASE     = "#EFEFE9"   # districts with no cotton
    BASE_EDGE = "#FFFFFF"
    COTTON   = "#9DB08A"   # cotton-producing districts
    OUTLIER  = "#C0392B"   # Chemba
    RULE     = "#DCDCD4"

    # ── Data ──────────────────────────────────────────────────────────────────────
    gdf = gpd.read_file(TZ_DISTRICTS_PATH)

    dist = pd.read_csv(os.path.join(RESULTS_TABLES_DIR,
                                    "district_scores_all_scenarios.csv"))
    cotton_ids = set(
        dist[(dist.scenario == "baseline") & (dist.category == "Land use occupation")]
        ["district_id"].unique()
    )

    water = dist[(dist.category == "Water consumption") &
                 (dist.scenario == "irrigation")]
    w_tot = water["score"].sum()
    w_chemba = water.loc[water.district_id == CHEMBA, "score"].sum()
    w_next = water[water.district_id != CHEMBA]["score"].max()
    share = 100 * w_chemba / w_tot
    ratio = w_chemba / w_next

    gdf["cls"] = "base"
    gdf.loc[gdf.ADM2_PCODE.isin(cotton_ids), "cls"] = "cotton"
    gdf.loc[gdf.ADM2_PCODE == CHEMBA, "cls"] = "outlier"

    # ── Figure ────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.6, 8.4), dpi=200)
    fig.patch.set_facecolor("white")

    gdf[gdf.cls == "base"].plot(ax=ax, facecolor=BASE, edgecolor=BASE_EDGE,
                                linewidth=0.4, zorder=1)
    gdf[gdf.cls == "cotton"].plot(ax=ax, facecolor=COTTON, edgecolor=BASE_EDGE,
                                  linewidth=0.4, zorder=2)
    gdf[gdf.cls == "outlier"].plot(ax=ax, facecolor=OUTLIER, edgecolor="#7B241C",
                                   linewidth=1.4, zorder=4)

    # national outline
    gdf.dissolve().boundary.plot(ax=ax, color="#8A8A82", linewidth=0.9, zorder=3)

    ax.set_axis_off()
    ax.set_aspect("equal")

    # ── Chemba callout ────────────────────────────────────────────────────────────
    c = gdf[gdf.ADM2_PCODE == CHEMBA].geometry.iloc[0]
    cx, cy = c.centroid.x, c.centroid.y

    x0, y0 = ax.get_xlim()
    x1, y1 = ax.get_ylim()
    lx, ly = cx + 3.4, cy + 2.6          # label anchor, up and to the right

    ax.add_patch(FancyArrowPatch((lx - 0.15, ly - 0.12), (cx + 0.15, cy + 0.1),
                                 arrowstyle="-", linewidth=1.1, color=INK,
                                 shrinkA=0, shrinkB=0, zorder=5))
    ax.plot([cx], [cy], marker="o", markersize=4.2, color=OUTLIER,
            markeredgecolor="white", markeredgewidth=0.9, zorder=6)

    ax.text(lx, ly, "Chemba District", fontsize=11.5, fontweight="bold",
            color=INK, ha="left", va="bottom", zorder=6)
    ax.text(lx, ly - 0.55, "Dodoma Region  ·  TZ0107", fontsize=9,
            color=MUTED, ha="left", va="top", zorder=6)

    # ── Title, legend, note ───────────────────────────────────────────────────────
    ax.set_title("Location of Chemba District, the water-consumption outlier",
                 fontsize=12.5, fontweight="bold", color=INK, pad=14, loc="left")

    handles = [
        mpatches.Patch(facecolor=OUTLIER, edgecolor="#7B241C", linewidth=1.0,
                       label="Chemba District (excluded)"),
        mpatches.Patch(facecolor=COTTON, edgecolor=BASE_EDGE,
                       label="Cotton-producing districts"),
        mpatches.Patch(facecolor=BASE, edgecolor=BASE_EDGE,
                       label="Other districts"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9.5,
                    handlelength=1.5, handleheight=1.0, borderpad=0.4,
                    labelspacing=0.6)
    for t in leg.get_texts():
        t.set_color(MUTED)

    note = (
        f"Chemba's PCR-GLOBWB basin characterisation factor is roughly "
        f"{ratio:,.0f}× that of the next-highest district. Under the irrigation\n"
        f"scenario it alone would account for {share:.1f}% of the national "
        f"water-consumption score, so it is excluded from district water totals\n"
        f"in all figures."
    )
    fig.text(0.02, 0.030, note, fontsize=8.6, color=MUTED, ha="left", va="bottom",
             linespacing=1.6)

    plt.tight_layout(rect=(0, 0.07, 1, 1))

    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
    for ext in ("png",):
        out = os.path.join(RESULTS_FIGURES_DIR, f"fig_S4.{ext}")
        plt.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"Saved -> {out}")
    plt.close()

    print(f"\nChemba share of irrigation water score: {share:.2f}%")
    print(f"Chemba / next-highest district ratio:   {ratio:,.0f}x")
    print(f"Cotton-producing districts mapped:      {len(cotton_ids)}")


def make_si_sankey():
    """SI - physical cotton flow sankey (plotly + kaleido).

    Integrated from make_sankey.py.
    """
    """
    make_sankey.py
    Sankey diagram of physical cotton flows (tonnes/year, baseline scenario).
    Waste streams from ginning, textile mfg. and oil milling combine into one
    'Waste & Disposal' node. Saves landscape PNG + 90°-rotated portrait PNG.
    """

    import plotly.graph_objects as go
    from PIL import Image
    import os

    OUT_DIR       = "X:/Eli/projects/tz_cotton/python/results/figures"
    OUT_HTML      = os.path.join(OUT_DIR, "fig_S2.html")
    OUT_PNG       = os.path.join(OUT_DIR, "fig_S2.png")
    OUT_PNG_VERT  = os.path.join(OUT_DIR, "fig_S2_vertical.png")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Annual volumes (tonnes/year, baseline scenario) ──────────────────────────
    RAW_COTTON_T    = 282_510   # 445,817 ha × 633.69 kg/ha (TCB 2024)

    LINT_T          = round(RAW_COTTON_T / 2.699561404)      # ~104,648 t
    SEED_T          = RAW_COTTON_T - LINT_T                  # ~177,862 t

    EXPORT_LINT_T   = 82_448
    LINT_TO_TEX_T   = round(17_159 * 1.201201)              # ~20,614 t lint consumed
    TEXTILE_T       = 17_159

    REPLANTING_T    = 25_000
    # All remaining seed goes to oil milling (no untracked residual)
    SEED_OIL_MILL_T = SEED_T - REPLANTING_T                         # ~152,862 t
    # Derive total oil and cake outputs from seed input
    # Waste is carved out of the seed input, so: oil + cake + waste = seed_input
    SEED_OIL_T      = round(SEED_OIL_MILL_T * 0.308411215 / 1.308411215)  # ~36,037 t
    OIL_WASTE_T     = round(SEED_OIL_T * 0.560748)                         # ~20,214 t
    SEED_CAKE_T     = SEED_OIL_MILL_T - SEED_OIL_T - OIL_WASTE_T          # ~96,611 t
    # Check: SEED_OIL_T + SEED_CAKE_T + OIL_WASTE_T == SEED_OIL_MILL_T  ✓

    # ── Waste streams (from inventory rates) ─────────────────────────────────────
    # Ginning: fibre/linter waste (short fibres, dust, motes)
    #   rate = 0.040619 kg waste / kg lint  (inventory row 31)
    GINNING_WASTE_T = round(LINT_T * 0.040619)              # ~4,251 t

    # Textile mfg.: cut-and-sew waste, yarn offcuts
    #   rate = 0.201201 kg waste / kg textile  (inventory row 57)
    TEXTILE_WASTE_T = round(TEXTILE_T * 0.201201)           # ~3,454 t

    # ── Nodes ────────────────────────────────────────────────────────────────────
    #  idx  label
    #   0   On-Farm Production
    #   1   Ginning
    #   2   Cotton Lint
    #   3   Cotton Seed
    #   4   Lint Export
    #   5   Textile Mfg.
    #   6   Oil Milling
    #   7   Seed Replanting
    #   8   Domestic Textile
    #   9   Cottonseed Oil
    #  10   Seed Cake
    #  11   Waste & Disposal        (ginning + textile + oil mill waste combined)

    NODE_LABELS = [
        "On-Farm<br>Production",          # 0
        "Ginning",                        # 1
        "Cotton Lint",                    # 2
        "Cotton Seed",                    # 3
        "Lint Export",                    # 4
        "Textile Mfg.",                   # 5
        "Oil Milling",                    # 6
        "Seed for<br>Replanting",         # 7
        "Domestic<br>Textile",            # 8
        "Cottonseed Oil",                 # 9
        "Seed Cake<br>(Livestock Feed)",  # 10
        "Waste &<br>Disposal",            # 11
    ]

    # ── Colours ──────────────────────────────────────────────────────────────────
    GREEN_D  = "#2D6A4F"
    GREEN_M  = "#52B788"
    GREEN_L  = "#95D5B2"
    BLUE_D   = "#1B4F72"
    ORANGE_D = "#A04000"
    ORANGE_M = "#E67E22"
    TEAL     = "#148F77"
    GRAY     = "#8D9EAB"
    GRAY_D   = "#5D6D7E"
    RED_GRAY = "#922B21"

    NODE_COLORS = [
        GREEN_D,    # 0 On-Farm Production
        GREEN_M,    # 1 Ginning
        GREEN_L,    # 2 Cotton Lint
        GREEN_L,    # 3 Cotton Seed
        BLUE_D,     # 4 Lint Export
        TEAL,       # 5 Textile Mfg.
        ORANGE_D,   # 6 Oil Milling
        GRAY,       # 7 Seed Replanting
        TEAL,       # 8 Domestic Textile
        ORANGE_M,   # 9 Cottonseed Oil
        "#C9860A",  # 10 Seed Cake
        RED_GRAY,   # 11 Waste & Disposal
    ]

    # ── Links ────────────────────────────────────────────────────────────────────
    links = [
        # (source, target, value, label, color)
        (0,  1, RAW_COTTON_T,    "Raw cotton to ginning",               "rgba(45,106,79,0.35)"),
        (1,  2, LINT_T,          "Cotton lint",                         "rgba(149,213,178,0.45)"),
        (1,  3, SEED_T,          "Cotton seed",                         "rgba(149,213,178,0.45)"),
        (2, 11, GINNING_WASTE_T, "Lint processing waste (fibres/motes)", "rgba(146,43,33,0.40)"),
        (2,  4, EXPORT_LINT_T,   "Lint exported internationally",       "rgba(27,79,114,0.40)"),
        (2,  5, LINT_TO_TEX_T,   "Lint to textile manufacturing",       "rgba(20,143,119,0.40)"),
        (3,  6, SEED_OIL_MILL_T, "Cottonseed to oil milling",           "rgba(160,64,0,0.40)"),
        (3,  7, REPLANTING_T,    "Seed retained for next season",       "rgba(141,158,171,0.40)"),
        (5,  8, TEXTILE_T,       "Domestic textile output",             "rgba(20,143,119,0.40)"),
        (5, 11, TEXTILE_WASTE_T, "Textile cut & process waste",         "rgba(146,43,33,0.40)"),
        (6,  9, SEED_OIL_T,      "Cottonseed oil",                      "rgba(230,126,34,0.40)"),
        (6, 10, SEED_CAKE_T,     "Seed cake for livestock feed",        "rgba(201,134,10,0.40)"),
        (6, 11, OIL_WASTE_T,     "Oil mill process & solid waste",      "rgba(146,43,33,0.40)"),
    ]

    links = [(s, t, v, l, c) for s, t, v, l, c in links if v > 0]

    # ── Node positions (arrangement="fixed") ─────────────────────────────────────
    #  x: column positions (0 = left, 1 = right)
    #  y: row positions   (0 = top,  1 = bottom)
    #  idx  node
    #   0   On-Farm Production
    #   1   Ginning
    #   2   Cotton Lint
    #   3   Cotton Seed
    #   4   Lint Export
    #   5   Textile Mfg.
    #   6   Oil Milling
    #   7   Seed Replanting
    #   8   Domestic Textile
    #   9   Cottonseed Oil
    #  10   Seed Cake
    #  11   Waste & Disposal
    # Col 1 (x=0.01): On-Farm Production
    # Col 2 (x=0.16): Ginning
    # Col 3 (x=0.32): Cotton Lint, Cotton Seed
    # Col 4 (x=0.56): Textile Mfg., Oil Milling   (intermediate processors only)
    # Col 5 (x=0.82): Lint Export, Seed Replanting, Domestic Textile,
    #                  Cottonseed Oil, Seed Cake, Waste & Disposal
    #         idx:  0     1     2     3     4     5     6     7     8     9    10    11
    NODE_X = [0.01, 0.16, 0.32, 0.32, 0.82, 0.56, 0.56, 0.82, 0.82, 0.82, 0.82, 0.82]
    # y positions derived proportionally from flow volumes so nodes tile without overlap.
    # Final-column reference total ≈ 285,174 t; each node center = cumulative_fraction + half_node_height.
    # Col 3: Cotton Lint 104,648 t / Cotton Seed 177,862 t of 282,510 t total.
    # Col 4 intermediate nodes aligned with their weighted output centres.
    #              0     1      2      3      4      5     6      7      8      9     10     11
    NODE_Y = [0.50, 0.50, 0.185, 0.685, 0.145, 0.35, 0.71, 0.393, 0.319, 0.500, 0.733, 0.951]

    # ── Build figure ─────────────────────────────────────────────────────────────
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color="white", width=0.5),
            label=NODE_LABELS,
            color=NODE_COLORS,
            x=NODE_X,
            y=NODE_Y,
            hovertemplate="%{label}<br><b>%{value:,.0f} t/yr</b><extra></extra>",
        ),
        link=dict(
            source=[s for s, t, v, l, c in links],
            target=[t for s, t, v, l, c in links],
            value= [v for s, t, v, l, c in links],
            label= [l for s, t, v, l, c in links],
            color= [c for s, t, v, l, c in links],
            hovertemplate="%{label}<br><b>%{value:,.0f} t/yr</b><extra></extra>",
        ),
    ))

    fig.update_layout(
        title=dict(
            text=(
                "<b>Physical Cotton Flows — Tanzania Cotton Supply Chain</b><br>"
                "<sup>Baseline scenario  |  Tonnes per year  |  "
                "445,817 ha × 633.69 kg/ha ≈ 282,510 t seed cotton</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=17, color="#1B2631"),
        ),
        font=dict(family="Arial", size=13, color="#1B2631"),
        paper_bgcolor="#FAFAFA",
        plot_bgcolor="#FAFAFA",
        width=1500,
        height=900,
        margin=dict(l=20, r=20, t=95, b=20),
    )

    # ── Save ─────────────────────────────────────────────────────────────────────
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"HTML saved    -> {OUT_HTML}")

    fig.write_image(OUT_PNG, scale=3)
    print(f"PNG landscape -> {OUT_PNG}")

    img = Image.open(OUT_PNG)
    img.rotate(-90, expand=True).save(OUT_PNG_VERT)
    print(f"PNG vertical  -> {OUT_PNG_VERT}")


def make_si_mc_plots(scenario="baseline"):
    """SI - Monte Carlo boxplot + relative-uncertainty plot for one scenario.

    Integrated from mc_plots.py.
    """
    import os as _os
    from src.config import RESULTS_TABLES_DIR as _RT
    _mc = [f for f in _os.listdir(_RT) if f.startswith('mc_')]
    if not _mc:
        print('make_si_mc_plots: no mc_* tables in results/tables '
              '- run 02_monte_carlo first. SKIPPED.')
        return
    """
    mc_plots.py
    Standalone reproduction of the two Monte Carlo plot cells from
    03_monte_carlo.ipynb. Reads the saved MC tables and writes both figures to
    results/figures/. No Brightway needed.

    Run:
        conda run -n bw25-regional python mc_plots.py            # baseline
        conda run -n bw25-regional python mc_plots.py high_yield # any scenario

    Outputs (in results/figures/):
        mc_boxplot_<scenario>.png              log-axis score distributions
        mc_relative_uncertainty_<scenario>.png normalised (comparable) boxplots
    """
    import sys, os

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.config import RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR

    # Category order (matches monte_carlo._METHODS)
    CATS = ["Land use occupation", "Water consumption", "FW eutrophication N",
            "FW eutrophication P", "Climate change rcp26", "FW ecotoxicity"]

    PLOT_SCENARIO = scenario

    long_df = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "mc_scores_all_scenarios.csv"))
    summ_df = pd.read_csv(os.path.join(RESULTS_TABLES_DIR, "mc_summary_all_scenarios.csv"))

    if PLOT_SCENARIO not in long_df["scenario"].unique():
        sys.exit(f"scenario '{PLOT_SCENARIO}' not in MC tables. "
                 f"Available: {sorted(long_df['scenario'].unique())}")

    N_ITERATIONS = int(long_df.groupby(["scenario", "category"]).size().iloc[0])
    os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)

    sub = long_df[long_df["scenario"] == PLOT_SCENARIO]
    cats = [c for c in CATS if c in sub["category"].unique()]
    data = [sub[sub["category"] == c]["score"].values for c in cats]
    det  = summ_df[summ_df["scenario"] == PLOT_SCENARIO].set_index("category")["deterministic"]
    cvv  = summ_df[summ_df["scenario"] == PLOT_SCENARIO].set_index("category")["cv_pct"]

    # FW ecotoxicity: substitute the stable-subset draws used for the
    # main-text uncertainty intervals. The full-inventory distribution is
    # dominated by ~40 sign-crossing background waste-treatment flows whose
    # variance is an artifact (raw CV ~81%, ~10% negative draws); the stable
    # subset excludes them at ~1.3% of the deterministic score. See
    # src/mc_ecotox.py and Methods.
    _stable_path = os.path.join(RESULTS_TABLES_DIR, "mc_ecotox_stable_draws.csv")
    if os.path.exists(_stable_path) and "FW ecotoxicity" in cats:
        _sd = pd.read_csv(_stable_path)
        _sd = _sd[_sd["scenario"] == PLOT_SCENARIO]
        if len(_sd):
            _idx = cats.index("FW ecotoxicity")
            data[_idx] = _sd["score_stable"].values
            _v = data[_idx]
            cvv = cvv.copy()
            cvv["FW ecotoxicity"] = 100.0 * _v.std() / _v.mean()
            print(f"make_si_mc_plots: FW ecotoxicity draws replaced with the "
                  f"stable subset (n={len(_v)}, CV={cvv['FW ecotoxicity']:.1f}%)")

    # ── Plot 1: log-axis score distributions ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.boxplot(data, tick_labels=[c.replace(" ", "\n", 1) for c in cats],
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#DDE7F0", edgecolor="#33526e"),
               medianprops=dict(color="#B22222", linewidth=1.5))
    for i, c in enumerate(cats, 1):
        ax.plot(i, det.get(c, np.nan), "D", color="#111", markersize=5, zorder=5)
    ax.set_yscale("log")
    ax.set_ylabel("Endpoint score (PDF·yr, log scale)", fontsize=9)
    ax.set_title(f"Monte Carlo score distributions — {PLOT_SCENARIO}\n"
                 f"box = IQR, red = MC median, black diamond = deterministic  "
                 f"(N={N_ITERATIONS})", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(RESULTS_FIGURES_DIR, f"mc_boxplot_{PLOT_SCENARIO}.png")
    plt.savefig(p1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {p1}")

    # ── Plot 2: normalised (comparable) distributions ─────────────────────────────
    # built from `data` (not re-read from sub) so the ecotoxicity
    # stable-subset substitution above carries through
    norm = [d_ / det[c] for d_, c in zip(data, cats)]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.axhline(1.0, color="#111", linestyle="--", linewidth=0.8, zorder=1,
               label="deterministic")
    ax.boxplot(norm, tick_labels=[c.replace(" ", "\n", 1) for c in cats],
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#DDE7F0", edgecolor="#33526e"),
               medianprops=dict(color="#B22222", linewidth=1.5))
    y_top = max(np.percentile(n, 97.5) for n in norm)
    ax.set_ylim(top=y_top * 1.20)          # headroom so CV labels clear the title
    for i, c in enumerate(cats, 1):
        ax.text(i, y_top * 1.04, f"CV {cvv[c]:.0f}%", ha="center", va="bottom",
                fontsize=7.5, color="#555")
    ax.set_ylabel("Score / deterministic value", fontsize=9)
    ax.set_title(f"Monte Carlo relative uncertainty — {PLOT_SCENARIO}\n"
                 f"each category normalised to its own deterministic value  "
                 f"(N={N_ITERATIONS})", fontsize=10, pad=14)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.6)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    p2 = os.path.join(RESULTS_FIGURES_DIR, f"mc_relative_uncertainty_{PLOT_SCENARIO}.png")
    plt.savefig(p2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {p2}")
