"""Shared LCA setup helpers used by both 01_scenarios.ipynb and monte_carlo.py.

Extracting spatial-weight preparation and demand construction here keeps the
deterministic notebook and the Monte Carlo script from drifting apart. The
logic mirrors 01_scenarios.ipynb cell 3 (spatial_data) and cell 7 (demand);
01 can be refactored to import these too, but does not have to be.
"""
import geopandas as gpd
gpd.options.io_engine = "fiona"

import bw2data as bd
import bw2regional as bwr

from src.config import (
    TZ_DISTRICTS_PATH, GINNERIES_PATH, TEXTILE_PLANTS_PATH, XT_COTTON_PRODUCTION,
)

# Functional-unit PRODUCT codes (see src/inventory.py _prod definitions).
# The LCA demand is keyed on products, not the processes that make them:
#   prod8 domestic_consumption  <- act8 domestic_market
#   prod7 foreign_consumption    <- act7 export_market
_CODE_DOMESTIC = "prod8"   # domestic_consumption basket
_CODE_FOREIGN  = "prod7"   # exported lint

_SKIP_COLS = {"index_right", "SymbolID", "AltMode", "Base", "HasLabel", "LabelID"}


def prepare_spatial_data():
    """Return the spatial_data dict required by build_foreground_db().

    Mirrors 01_scenarios.ipynb cell 3: production weights from the cotton
    ExtensionTable, ginning weights from ginnery production joined to districts,
    textile weights from plant counts per district. Ginning/textile fall back
    to production weights when their source data is unusable.
    """
    tz_districts_gdf = gpd.read_file(TZ_DISTRICTS_PATH)
    ginneries_gdf    = gpd.read_file(GINNERIES_PATH)
    textile_gdf      = gpd.read_file(TEXTILE_PLANTS_PATH)

    # Production weights (SPAM 2020 via ExtensionTable)
    xt = bwr.ExtensionTable(XT_COTTON_PRODUCTION)
    xtable = xt.load()
    prod_by_district = {district_id: value for value, (_, district_id) in xtable}
    total_prod = sum(prod_by_district.values())
    production_weights = {d: v / total_prod for d, v in prod_by_district.items()}

    # Ginning weights
    gin_proj = ginneries_gdf.to_crs(tz_districts_gdf.crs)
    gin_join = gpd.sjoin(
        gin_proj, tz_districts_gdf[["ADM2_PCODE", "ADM2_EN", "geometry"]],
        how="left", predicate="within",
    )
    if "Production" in gin_join.columns:
        gin_col = "Production"
    else:
        _num = gin_join.select_dtypes(include=[float, int]).columns.tolist()
        _cand = [c for c in _num if c not in _SKIP_COLS]
        gin_col = _cand[0] if _cand else None
    if gin_col and gin_col in gin_join.columns:
        agg = gin_join.groupby("ADM2_PCODE")[gin_col].sum()
        tot = agg.sum()
        ginning_weights = ({d: v / tot for d, v in agg.items() if v > 0}
                           if tot > 0 else dict(production_weights))
    else:
        dists = gin_join["ADM2_PCODE"].dropna().unique()
        ginning_weights = ({d: 1 / len(dists) for d in dists}
                           if len(dists) else dict(production_weights))

    # Textile weights (plant counts)
    tex_proj = textile_gdf.to_crs(tz_districts_gdf.crs)
    tex_join = gpd.sjoin(
        tex_proj, tz_districts_gdf[["ADM2_PCODE", "ADM2_EN", "geometry"]],
        how="left", predicate="within",
    )
    if len(tex_join) and "ADM2_PCODE" in tex_join.columns:
        counts = tex_join["ADM2_PCODE"].dropna().value_counts()
        tot = counts.sum()
        textile_weights = ({d: c / tot for d, c in counts.items()}
                           if tot > 0 else dict(production_weights))
    else:
        textile_weights = dict(production_weights)

    return {
        "tz_districts_gdf":   tz_districts_gdf,
        "production_weights": production_weights,
        "ginning_weights":    ginning_weights,
        "textile_weights":    textile_weights,
    }


def build_demand(scenario, foreground_db="foreground"):
    """Return the functional-unit demand dict for a scenario.

    Mirrors 01_scenarios.ipynb cell 7: one year of domestic industry output
    (domestic consumption basket + exported lint).
    """
    db = bd.Database(foreground_db)
    domestic = next(a for a in db if a["code"] == _CODE_DOMESTIC)
    foreign  = next(a for a in db if a["code"] == _CODE_FOREIGN)
    return {
        domestic.id: scenario.domestic_total_t(),
        foreign.id:  scenario.export_lint_t(),
    }
