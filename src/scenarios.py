"""
scenarios.py — Scenario definitions for the Tanzania Cotton LCA + DLS study.

A Scenario bundles:
    (1) per-unit LCA exchange amounts (consumed by inventory.py), and
    (2) DLS parameter dataclasses (consumed by needs.py via run_dls()).

The two sides share a common `yield_kg_ha` and `organic_share` so that a
single scenario object fully describes both an LCA run and a DLS roll-up.

PARAMETER DERIVATION
--------------------
Conventional/organic per-ha application rates come from the Parameters
sheet of cotton_industry_data.xlsx (TCB 2024). Per-kg LCA inputs are
derived from the Scenarios sheet of the same workbook, using:
    rate_per_kg = (organic_share*org_rate_ha + (1-organic_share)*conv_rate_ha)
                  / yield_kg_ha

Farmgate price is similarly blended from conventional ($0.46/kg) and
organic ($0.529/kg ≈ 15% bioRe Tanzania premium).

For each scenario, the derivation is shown inline so the numeric
literals below can be reproduced by hand.

SOURCE-ROW MAP (cotton_industry_data.xlsx::Parameters)
------------------------------------------------------
    row 53  Yield per ha               633.69 kg/ha     (TCB 2024 calculated)
    row 54  Yield per ha (target)      1,400 kg/ha      (TCDS national target)
    row 59  Drought crop failure rate  0.6
    row 61  Organic production rate    0.40 (2024)
    row 73  Yield per ha (current)     400 kg/ha        (farmer-reported)
    row 83  Nitrogen application       7.0 kg N/ha      (conventional)
    row 84  Nitrogen application org   28.3 kg N/ha
    row 85  P2O5 application org       14.1 kg/ha       (no conv P rate)
    Pesticide applications     7/yr (TCB 2024; the 4.5 ± 1.5 farmer-
                               interview figure is the uncertainty anchor)
    Pesticide weight ratio     0.034 kg/acre/spray
    NOTE: sheet row numbers drift as rows are added/removed — match
    entries by Parameter name, not row index.
    row 87  Biopesticide application   0.016 kg/ha      (organic)

The current-state baseline yield is 633.69 kg/ha (TCB 2024 calculated
from raw cotton output / planted area, FAOSTAT-corroborated). The
old farmer-reported 400 kg/ha is retained as a low-yield reference but
is no longer the active baseline — every scenario below uses 633.69 as
its anchor yield (or a multiple/fraction thereof).
"""
from __future__ import annotations
from dataclasses import dataclass, field

from src.needs import (
    NutritionParams,
    ClothingParams,
    FarmIncomeParams,
    WageParams,
    EmploymentParams,
    NationalContext,
    assess_needs_from_scenario,
)

# ═════════════════════════════════════════════════════════════════════════════
# APPLICATION-RATE CONSTANTS (cotton_industry_data.xlsx::Parameters)
# ═════════════════════════════════════════════════════════════════════════════

# Nitrogen (kg N / ha)
N_RATE_CONV_KG_HA = 7.0      # row 83
N_RATE_ORG_KG_HA  = 28.3     # row 84 (manure-driven; higher mass, lower N avail)

# Phosphorus (kg P2O5 / ha) — conventional assumed zero per user
P_RATE_CONV_KG_HA = 0.0
P_RATE_ORG_KG_HA  = 14.1     # row 85

# Pesticide (kg active ingredient / ha)
# Conventional: 7 applications/yr (TCB 2024, 'probably more reliable') ×
#               0.034 kg/acre/spray × 2.471 acre/ha = 0.588 kg/ha.
# The farmer-interview figure of 4.5 ± 1.5 applications/yr is retained in
# the Parameters sheet as the uncertainty anchor. The scenario literals
# below (profenofos_kg etc.) come from the Scenarios sheet and are built
# on the 7-application rate.
PEST_RATE_CONV_KG_HA = 7 * 0.034 * 2.471   # = 0.588
PEST_RATE_ORG_KG_HA  = 0.016                 # row 87 (biopesticide)

# Pesticide AI split (registered cotton EC co-formulation in E/W Africa):
#   Profenofos 300 g/L + Lambda-cyhalothrin 15 g/L → 95.0:5.0 by mass
PEST_PROF_FRAC   = 0.95
PEST_LAMBDA_FRAC = 0.05

# Yield anchors (kg seed cotton / ha)
YIELD_BASELINE_KG_HA    = 633.69      # TCB 2024 calculated (Parameters row 53)
YIELD_TARGET_KG_HA      = 2471.0      # TCDS national target (Parameters row 54; updated)
YIELD_IRRIGATION_KG_HA  = 1430.237547 # India national average irrigated yield (Scenarios sheet J2; WFN Report 68)

# P2O5 -> elemental P mass conversion = (2 x 30.974) / 141.943.
# Fertilizer P is conventionally reported as P2O5 (Parameters rows 88/90), and
# the ecoinvent input "nutrient supply from manure, solid, cattle" has the
# reference product "organic phosphorus fertiliser, as P2O5" — so fertilizer_p_kg
# stays in P2O5 and needs no conversion. The BIOSPHERE flow is different: it is
# "Phosphorus (to surface water)", and METHOD_P_EUTRO is registered with
# unit="PDF*yr/kg_P", i.e. per kg of ELEMENTAL P. p_emission_kg must therefore be
# converted. Note the N side needs no equivalent factor: fertilizer N is already
# reported as elemental N and METHOD_N_EUTRO is registered per kg_N.
P2O5_TO_P = 0.436421

# Baseline national cultivated area (ha). Mirrors NationalContext.national_area_ha
# and is the reference against which a scenario's area change is measured.
BASELINE_AREA_HA = 445_817   # = 282,510 t / 633.69 kg/ha; consistent with the
                             # Parameters-sheet planted area and NationalContext.
                             # (706,000 ha belonged to the 400-kg/ha pair.)


# Farmgate price (USD / kg seed cotton) — blended values from Scenarios sheet D14/F14
PRICE_CONV_USD_KG = 0.46      # TCB/TCDB indicative, 2022-23
PRICE_ORG_USD_KG  = 0.529     # +15% bioRe Tanzania premium; Kipepeo / Remei 2023-24
                              # https://www.biore.ch/en/news/farmers-interviewed-by-mr-simon-hohmann-co-ceo-remei-ag/

# Input cost (USD / kg seed cotton) — Scenarios sheet row 16
INPUT_COST_CONV_USD_KG = 0.0884   # baseline / high_yield / mfg / irrigation
INPUT_COST_ORG_USD_KG  = 0.0821   # organic_expansion / organic_high_yield


# ═════════════════════════════════════════════════════════════════════════════
# BLENDING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def blend_n_kg_per_kg(yield_kg_ha: float, organic_share: float) -> float:
    """Blended nitrogen application, kg N per kg raw cotton."""
    n_ha = organic_share * N_RATE_ORG_KG_HA + (1 - organic_share) * N_RATE_CONV_KG_HA
    return n_ha / yield_kg_ha


def blend_p_kg_per_kg(yield_kg_ha: float, organic_share: float) -> float:
    """Blended P2O5 application, kg P per kg raw cotton."""
    p_ha = organic_share * P_RATE_ORG_KG_HA + (1 - organic_share) * P_RATE_CONV_KG_HA
    return p_ha / yield_kg_ha


def blend_pesticide_kg_per_kg(yield_kg_ha: float, organic_share: float) -> float:
    """Blended pesticide application, kg AI per kg raw cotton.

    Biopesticide (organic) and conventional synthetic pesticide are summed
    into one flow; both are currently mapped to ecoinvent 'pesticide,
    unspecified' — an over-estimate for the organic portion, flagged as a
    conservative simplification pending a biopesticide-specific flow.
    """
    p_ha = organic_share * PEST_RATE_ORG_KG_HA + (1 - organic_share) * PEST_RATE_CONV_KG_HA
    return p_ha / yield_kg_ha


def blend_farmgate_price(organic_share: float) -> float:
    """Blended farmgate price in USD per kg seed cotton."""
    return organic_share * PRICE_ORG_USD_KG + (1 - organic_share) * PRICE_CONV_USD_KG


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO DATACLASS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """Unified scenario holding LCA exchange amounts AND DLS parameters.

    All LCA amounts are per kg of the reference product of each activity
    (raw cotton, lint, seed, textile, cottonseed) unless otherwise noted.
    All DLS parameters follow the schema in needs.py.

    No auto-propagation: every field is directly editable. Derived
    values are computed once when a scenario is defined and hard-coded
    as literals, with the formula shown inline so reviewers can
    reproduce them by hand.
    """
    name: str
    description: str = ""

    # ── Shared drivers (read by both LCA and DLS sides) ──────────────────────
    yield_kg_ha:   float = YIELD_BASELINE_KG_HA   # 633.69 (Parameters row 53)
    organic_share: float = 0.40
    cotton_share_of_clothing: float = 0.5         # DLS clothing threshold scaling

    # ── LCA: cotton farming (per kg raw cotton output) ───────────────────────
    # Default values reproduce BASELINE (yield 633.69, organic 0.40)
    land_occupation_m2: float = round(10_000 / YIELD_BASELINE_KG_HA, 6)   # 15.7806
    seed_replant_kg:    float = 0.0885

    # Pesticide AI breakdown (conventional only — biopesticide carries a
    # separate, much lower toxicity profile and is currently not split out).
    # baseline = 0.95 × 0.378 / 633.69 + 0.05 × 0.378 / 633.69 (organic share zero on conv pesticide)
    # …but the actual values below come straight from Scenarios sheet rows 10-11.
    pesticide_kg:           float = 0.00052899 + 2.7842e-5   # = 5.5683e-4
    profenofos_kg:          float = 0.00052899      # Scenarios sheet D10
    lambda_cyhalothrin_kg:  float = 2.7842e-5       # Scenarios sheet D11

    # N / P emissions and fertilizer technosphere amounts
    n_emission_kg:   float = 0.024491   # Scenarios sheet D8 — total N to water (eutrophication flow)
    p_emission_kg:   float = P2O5_TO_P * 0.008900   # Scenarios sheet D9, converted P2O5 -> elemental P
    # N fertilizer split by source — used directly in inventory.py for GHG calculations
    # (N₂O EFs differ: 1.0 % synthetic vs 0.6 % organic; CO₂ only from urea hydrolysis)
    # Baseline derivation (organic_share=0.40, yield=633.69 kg/ha):
    #   fertilizer_n_kg_manure = 0.40 × 28.3 / 633.69 = 0.017864
    #   fertilizer_n_kg_urea   = 0.60 × 7.0  / 633.69 = 0.006627
    fertilizer_n_kg_manure: float = 0.017864   # organic (manure) N, kg per kg raw cotton
    fertilizer_n_kg_urea:   float = 0.006627   # synthetic (urea) N, kg per kg raw cotton
    fertilizer_p_kg: float = 0.008900   # blended P (organic only)

    tractor_kg:      float = 1.4159e-06      # tractors per kg raw cotton (count fraction)
    tractor_mass_kg: float = 2000.0          # mass per tractor (ecoinvent activity is per kg of tractor)
    irrigation_water_m3: float = 0.0         # irrigation water withdrawal, m³/kg raw cotton (zero in non-irrigation scenarios)

    # ── LCA: ginning (per kg input before allocation) ────────────────────────
    ginning_land_m2:         float = 0.003955379
    ginning_electricity_kwh: float = 0.004631733
    ginning_transport_tkm:   float = 537.2354109
    ginning_waste_kg:        float = 0.041666667
    ginning_building_m2:     float = 0.00131846

    # ── LCA: textile manufacturing (per kg textile output) ───────────────────
    textile_lint_kg:         float = 1.201201201
    textile_land_m2:         float = 0.062794521
    textile_water_air_m3:    float = 0.02625
    textile_water_river_m3:  float = 0.0875
    textile_electricity_kwh: float = 6.575342466
    textile_coal_kwh:        float = 0.794277142
    textile_building_m2:     float = 0.020931507
    textile_starch_kg:       float = 0.037
    textile_waste_kg:        float = 0.201201201
    textile_wastewater_m3:   float = 0.06125

    # ── LCA: oil milling (per kg cottonseed input before allocation) ─────────
    oil_seed_input_kg:   float = 1.869158879
    oil_water_air_m3:    float = 6.72675e-08
    oil_water_river_m3:  float = 4.4845e-07
    oil_electricity_kwh: float = 0.060362851
    oil_mill_unit:       float = 3.41122e-07
    oil_waste_kg:        float = 0.560747664
    oil_wastewater_m3:   float = 3.81182e-07

    # ── LCA: export market ───────────────────────────────────────────────────
    export_transport_tkm: float = 3.72e1

    # ── DLS parameters (each a separately editable dataclass instance) ──────
    nutrition:   NutritionParams  = field(default_factory=NutritionParams)
    clothing:    ClothingParams   = field(default_factory=ClothingParams)
    farm_income: FarmIncomeParams = field(default_factory=FarmIncomeParams)
    wage:        WageParams       = field(default_factory=WageParams)
    employment:  EmploymentParams = field(default_factory=EmploymentParams)
    national:    NationalContext  = field(default_factory=NationalContext)

    # ── Domestic textile output (absolute lever) ────────────────────────────
    # Total domestic textile output, in tonnes/yr.  This is the ACTIVE lever
    # for the textile industry: setting it explicitly tells the model how
    # many tonnes of domestic fabric are produced.  When None, falls back to
    # the baseline value (`_baseline_textile_t`).  The implied
    # processing-share is derived downstream as
    #   share = domestic_textile_t × lint_per_fabric / total_lint_t
    # — a per-scenario observed quantity, not a control.
    domestic_textile_t_override: float | None = None

    # ── National-volume scaling anchors (TCB 2024 baseline) ──────────────────
    # These mirror the constants in config.py and are intentionally duplicated
    # here so a Scenario is self-contained.  Updated to the 633.69 kg/ha
    # baseline (raw cotton output 282,510 t/yr; Equations sheet row 1).
    _baseline_yield_kg_ha:    float = YIELD_BASELINE_KG_HA   # 633.69
    _baseline_seed_cake_t:    float = 80_333.440954
    _baseline_seed_oil_t:     float = 24_775.734126
    _baseline_textile_t:      float = 17_159.408311           # Scenarios sheet D14
    _baseline_export_lint_t:  float = 82_447.607501           # Scenarios sheet D13
    _textile_lint_per_fabric: float = 1.201201201             # kg lint / kg fabric (= textile_lint_kg)

    # ── Convenience ─────────────────────────────────────────────────────────
    def run_dls(self):
        """Run the full DLS assessment using this scenario's parameters."""
        return assess_needs_from_scenario(self)

    def national_production_t(self) -> float:
        """Derived: total cotton output at this yield × fixed national area."""
        return self.yield_kg_ha * self.national.national_area_ha / 1000.0

    # ── Scenario-scaled annual throughput (tonnes/yr) ────────────────────────
    # Drives the LCA functional-unit / demand vector so impacts scale
    # correctly with each lever:
    #   • Yield raises seed-cotton output, which raises ginning throughput
    #     and (downstream) oil-milling tonnage and total lint produced.
    #   • domestic_textile_t_override is the textile-industry lever: set it
    #     to raise domestic textile output independent of yield.  Surplus
    #     lint from yield gains spills over to export.  The processing
    #     share is now a derived (observed) quantity, not a control.
    # ─────────────────────────────────────────────────────────────────────────
    def _production_factor(self) -> float:
        """Total seed-cotton output relative to baseline.

        Production = yield x area, so BOTH levers scale the demand vector.
        This previously used yield alone, which meant a change in area scaled
        national_production_t() but NOT the downstream throughputs (seed cake,
        oil, lint) that drive the LCA demand. For any scenario at the baseline
        area the value is identical to the old yield-only factor, so existing
        scenario results are unchanged.
        """
        return ((self.yield_kg_ha * self.national.national_area_ha)
                / (self._baseline_yield_kg_ha * BASELINE_AREA_HA))

    def domestic_seed_cake_t(self) -> float:
        """National seed-cake output, scales with yield."""
        return self._baseline_seed_cake_t * self._production_factor()

    def domestic_seed_oil_t(self) -> float:
        """National seed-oil output, scales with yield."""
        return self._baseline_seed_oil_t * self._production_factor()

    def domestic_textile_t(self) -> float:
        """Domestic textile output (tonnes/yr).  Uses the per-scenario
        override if set, else the baseline value (capacity unchanged)."""
        if self.domestic_textile_t_override is not None:
            return self.domestic_textile_t_override
        return self._baseline_textile_t

    def total_lint_t(self) -> float:
        """Total ginned lint, scales with yield."""
        baseline_lint_t = (self._baseline_textile_t * self._textile_lint_per_fabric
                           + self._baseline_export_lint_t)
        return baseline_lint_t * self._production_factor()

    def export_lint_t(self) -> float:
        """Lint exported = total lint − lint absorbed by domestic textile."""
        domestic_lint_t = self.domestic_textile_t() * self._textile_lint_per_fabric
        return self.total_lint_t() - domestic_lint_t

    def effective_processing_share(self) -> float:
        """Observed share of national lint going to domestic textile.
        Derived from the absolute textile-output lever; not a control."""
        total = self.total_lint_t()
        if total <= 0:
            return 0.0
        return self.domestic_textile_t() * self._textile_lint_per_fabric / total

    def domestic_total_t(self) -> float:
        """Total domestic-market basket throughput (cake + oil + textile)."""
        return (self.domestic_seed_cake_t()
                + self.domestic_seed_oil_t()
                + self.domestic_textile_t())

    def domestic_basket_fractions(self) -> tuple[float, float, float]:
        """(cake, oil, textile) shares of the scenario-specific domestic basket."""
        total = self.domestic_total_t()
        return (
            self.domestic_seed_cake_t() / total,
            self.domestic_seed_oil_t()  / total,
            self.domestic_textile_t()   / total,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════
# All numeric values pulled from cotton_industry_data.xlsx::Scenarios.
# The `domestic_processing_share` field is a LEVER (intent), not an
# observed share — for scenarios that change yield without expanding mill
# capacity it stays at the baseline 0.20, even though the observed share
# in the Excel sheet drops (e.g. 0.0905 in high_yield) because total lint
# went up. The scaling logic above (`_dom_proc_factor`) correctly keeps
# domestic_textile_t fixed at 17,159 t/yr in those cases.
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE — current state, 40 % organic, no irrigation
# ─────────────────────────────────────────────────────────────────────────────
# Derivations (yield=633.69, organic_share=0.40):
#   N per ha       = 0.40*28.3 + 0.60*7.0 = 15.52 kg/ha
#   fertilizer_n_kg = 15.52 / 633.69 = 0.024491
#   P per ha       = 0.40*14.1 + 0.60*0.0 = 5.64 kg/ha
#   fertilizer_p_kg = 5.64 / 633.69 = 0.008900
#   Pesticide per ha = 0.40*0.016 + 0.60*0.378 = 0.2332 kg/ha
#   pesticide_kg   = 0.2332 / 633.69 = 3.681e-4   (this is the BLEND incl. biopesticide)
#   profenofos / λ-cyh come from synthetic pesticides only (Scenarios sheet
#   uses 95:5 split applied to conventional-share rate), giving:
#     profenofos_kg = 0.00052899
#     lambda_cyhalothrin_kg = 2.7842e-5
#   farmgate_price = 0.40*0.529 + 0.60*0.46 = 0.4784 USD/kg
#   input_cost_usd_per_ha = 0.0884 USD/kg × 633.69 = 56.0 USD/ha
# ─────────────────────────────────────────────────────────────────────────────
BASELINE = Scenario(
    name="baseline",
    description=(
        "Tanzania cotton current-state baseline — yield 633.69 kg/ha "
        "(TCB 2024 calculated), 40% organic share, 20% of lint processed "
        "domestically into textile."
    ),
    yield_kg_ha=YIELD_BASELINE_KG_HA,                  # 633.69
    organic_share=0.40,
    land_occupation_m2=10_000/YIELD_BASELINE_KG_HA,   # = 15.7806
    fertilizer_n_kg_manure=0.017864,                   # 0.40 × 28.3 / 633.69
    fertilizer_n_kg_urea=0.006627,                     # 0.60 × 7.0  / 633.69
    fertilizer_p_kg=0.008900,
    pesticide_kg=0.00052899 + 2.7842e-5,               # = 5.568e-4
    profenofos_kg=0.00052899,
    lambda_cyhalothrin_kg=2.7842e-5,
    n_emission_kg=0.024491,
    p_emission_kg=P2O5_TO_P * 0.008900,   # 0.008900 kg P2O5 -> 0.003884 kg P
    farm_income=FarmIncomeParams(
        farmgate_price_usd_per_kg_sc=0.4784,           # Scenarios sheet D14
        input_cost_usd_per_ha=INPUT_COST_CONV_USD_KG * YIELD_BASELINE_KG_HA,   # 56.0
    ),
    employment=EmploymentParams(
        farm_person_days_per_ha=120.0,                 # Scenarios sheet D17
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. HIGH_YIELD — TCDS national target yield (2,471 kg/ha)
# ─────────────────────────────────────────────────────────────────────────────
# Yield: 2,471 kg/ha is the updated TCB/TCDB national production target
#   (Parameters row 54), ~3.90× the 633.69 current baseline.
# Scaling rule: per-ha application rates scale linearly with yield →
#   per-kg rates IDENTICAL to BASELINE.
#     fertilizer_n_kg = 0.024491   (per-ha rate scales ×3.90)
#     fertilizer_p_kg = 0.008900
#     profenofos_kg   = 0.00052899
# Mill capacity unchanged → domestic_textile_t stays at 17,159 t/yr; the
#   extra lint goes to export (observed share collapses to ~0.051).
# Labour: 120 person-days × (2471/633.69) = 467.93 person-days/ha.
# input_cost_usd_per_ha = 0.0884 × 2471 = 218.4 USD/ha (per-kg cost held).
# ─────────────────────────────────────────────────────────────────────────────
HIGH_YIELD = Scenario(
    name="high_yield",
    description=(
        "Yield rises to the updated national target of 2,471 kg/ha (TCDS); "
        "organic share unchanged at 40%. Per-ha fertilizer/pesticide rates "
        "scale linearly with yield, so per-kg LCA inputs match BASELINE. "
        "National production rises to ~1,101,614 t/yr; mill capacity unchanged "
        "so the extra lint is exported. Higher yield drives 3.90× per-ha "
        "labour demand (~468 person-days)."
    ),
    yield_kg_ha=YIELD_TARGET_KG_HA,                    # 2471
    organic_share=0.40,
    land_occupation_m2=10_000/YIELD_TARGET_KG_HA,       # = 4.047
    fertilizer_n_kg_manure=0.017864,                   # per-ha scales with yield → same per-kg as BASELINE
    fertilizer_n_kg_urea=0.006627,                     # per-ha scales with yield -> same per-kg as BASELINE
    fertilizer_p_kg=0.008900,
    pesticide_kg=0.00052899 + 2.7842e-5,
    profenofos_kg=0.00052899,
    lambda_cyhalothrin_kg=2.7842e-5,
    n_emission_kg=0.024491,
    p_emission_kg=P2O5_TO_P * 0.008900,   # 0.008900 kg P2O5 -> 0.003884 kg P
    # Tractor passes (ploughing, spraying, weeding) are per-HECTARE field
    # operations set by agronomy, not by yield, so they are held fixed per-ha
    # rather than following the per-ha-scales-with-yield rule used above for
    # fertilizer and pesticide. Without this, high yield would imply 3.90x more
    # tractor passes per hectare.
    # NOTE: seed_replant_kg is deliberately LEFT at the baseline per-kg value
    # here (unlike IRRIGATION, where it is held per-ha).
    tractor_kg=1.4159e-06 * (YIELD_BASELINE_KG_HA / YIELD_TARGET_KG_HA),  # → 3.6311e-07
    farm_income=FarmIncomeParams(
        farmgate_price_usd_per_kg_sc=0.4784,
        input_cost_usd_per_ha=INPUT_COST_CONV_USD_KG * YIELD_TARGET_KG_HA,    # 218.4
    ),
    employment=EmploymentParams(
        farm_person_days_per_ha=467.926214,             # 120 × (2471/633.69); Scenarios sheet E17
    ),
    # domestic_textile_t_override left unset: mill capacity unchanged → 17,159 t
    # observed share collapses to ~0.051 because total lint rises with yield
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ORGANIC_EXPANSION — organic share doubles, yield unchanged
# ─────────────────────────────────────────────────────────────────────────────
# Derivations (yield=633.69, organic_share=0.80):
#   N per ha       = 0.80*28.3 + 0.20*7.0 = 24.04 kg/ha
#   fertilizer_n_kg = 24.04 / 633.69 = 0.037937
#   P per ha       = 0.80*14.1 + 0.20*0.0 = 11.28 kg/ha
#   fertilizer_p_kg = 11.28 / 633.69 = 0.017801
#   Pesticide per ha = 0.80*0.016 + 0.20*0.378 = 0.0884 kg/ha
#     (mostly biopesticide; synthetic AIs scale with conventional share only)
#     profenofos_kg          = 0.00017633   (Scenarios sheet F10)
#     lambda_cyhalothrin_kg  = 9.281e-6     (Scenarios sheet F11)
#   farmgate_price = 0.80*0.529 + 0.20*0.46 = 0.4968 USD/kg
#   input_cost_usd_per_ha = 0.0821 USD/kg × 633.69 = 52.0 USD/ha
# ─────────────────────────────────────────────────────────────────────────────
ORGANIC_EXPANSION = Scenario(
    name="organic_expansion",
    description=(
        "Organic share 40% → 80% via policy push; yield held at 633.69 kg/ha "
        "(no yield penalty per Meatu field trials, Springer 2020). Synthetic "
        "pesticide use drops to ~30% of baseline."
    ),
    yield_kg_ha=YIELD_BASELINE_KG_HA,                  # 633.69
    organic_share=0.80,
    land_occupation_m2=10_000/YIELD_BASELINE_KG_HA,
    fertilizer_n_kg_manure=0.035724,                   # 0.80 × 28.3 / 633.69
    fertilizer_n_kg_urea=0.002209,                     # 0.20 × 7.0  / 633.69
    fertilizer_p_kg=0.017801,
    pesticide_kg=0.00017633 + 9.281e-6,                # = 1.856e-4
    profenofos_kg=0.00017633,
    lambda_cyhalothrin_kg=9.281e-6,
    n_emission_kg=0.037937,
    p_emission_kg=P2O5_TO_P * 0.017801,   # 0.017801 kg P2O5 -> 0.007768 kg P
    farm_income=FarmIncomeParams(
        farmgate_price_usd_per_kg_sc=0.4968,           # Scenarios sheet F14
        input_cost_usd_per_ha=INPUT_COST_ORG_USD_KG * YIELD_BASELINE_KG_HA,   # 52.0
    ),
    employment=EmploymentParams(
        farm_person_days_per_ha=120.0,
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ORGANIC_HIGH_YIELD — target yield + organic expansion (DISABLED)
# ─────────────────────────────────────────────────────────────────────────────
# Same derivations as ORGANIC_EXPANSION but with yield=1,400 (per-kg rates
# identical to ORGANIC_EXPANSION; per-ha rates are 2.21× higher). Currently
# commented out — flip into ALL_SCENARIOS to enable.
# ─────────────────────────────────────────────────────────────────────────────
# ORGANIC_HIGH_YIELD = Scenario(
#     name="organic_high_yield",
#     description=(
#         "Combined: organic share 80% and yield 1,400 kg/ha (national "
#         "target). Per-ha rates scale linearly with yield; national "
#         "production rises to ~624,144 t/yr."
#     ),
#     yield_kg_ha=YIELD_TARGET_KG_HA,
#     organic_share=0.80,
#     land_occupation_m2=round(10_000 / YIELD_TARGET_KG_HA, 6),
#     fertilizer_n_kg_manure=0.035724,                 # same per-kg as ORGANIC_EXPANSION
#     fertilizer_n_kg_urea=0.002209,
#     fertilizer_p_kg=0.017801,
#     pesticide_kg=0.00017633 + 9.281e-6,
#     profenofos_kg=0.00017633,
#     lambda_cyhalothrin_kg=9.281e-6,
#     n_emission_kg=0.037937,
#     p_emission_kg=0.017801,
#     farm_income=FarmIncomeParams(
#         farmgate_price_usd_per_kg_sc=0.4968,
#         input_cost_usd_per_ha=INPUT_COST_ORG_USD_KG * YIELD_TARGET_KG_HA,   # 114.83
#     ),
#     employment=EmploymentParams(
#         farm_person_days_per_ha=265.114,
#     ),
# )


# ─────────────────────────────────────────────────────────────────────────────
# 5. MANUFACTURING_EXPANSION — domestic processing share 20% → 60%
# ─────────────────────────────────────────────────────────────────────────────
# Same farming inputs as BASELINE (yield 633.69, organic 0.40). Industrial-
# policy lever: triple the share of national lint that is locally processed
# into textile (0.20 → 0.60). Per-kg LCA inputs unchanged; total domestic
# textile output 17,159 t → 51,478 t; textile mill employment scales 3×.
# ─────────────────────────────────────────────────────────────────────────────
MANUFACTURING_EXPANSION = Scenario(
    name="manufacturing_expansion",
    description=(
        "Domestic textile output triples (17 → 51 kt/yr) via mill expansion; "
        "yield and organic share held at baseline values. Industrial-policy "
        "lever — sets domestic_textile_t_override directly. Observed "
        "processing share rises from 0.20 to 0.60 as a consequence, but the "
        "share is not the control."
    ),
    yield_kg_ha=YIELD_BASELINE_KG_HA,
    organic_share=0.40,
    land_occupation_m2=10_000/YIELD_BASELINE_KG_HA,
    fertilizer_n_kg_manure=0.017864,                   # same farming as BASELINE
    fertilizer_n_kg_urea=0.006627,
    fertilizer_p_kg=0.008900,
    pesticide_kg=0.00052899 + 2.7842e-5,
    profenofos_kg=0.00052899,
    lambda_cyhalothrin_kg=2.7842e-5,
    n_emission_kg=0.024491,
    p_emission_kg=P2O5_TO_P * 0.008900,   # 0.008900 kg P2O5 -> 0.003884 kg P
    domestic_textile_t_override=3*17_159.408311,     # = 51,478.225  (Scenarios H14)
    clothing=ClothingParams(
        domestic_fabric_kg_yr=3*17_159_408.3,        # 3× baseline textile output (kg)
    ),
    farm_income=FarmIncomeParams(
        farmgate_price_usd_per_kg_sc=0.4784,
        input_cost_usd_per_ha=INPUT_COST_CONV_USD_KG * YIELD_BASELINE_KG_HA,
    ),
    employment=EmploymentParams(
        farm_person_days_per_ha=120.0,
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MANUFACTURING_HIGH_YIELD — 60% domestic processing × yield 1,400 (DISABLED)
# ─────────────────────────────────────────────────────────────────────────────
# HIGH_YIELD farming side combined with the 60% domestic-processing lever.
# Currently commented out — flip into ALL_SCENARIOS to enable. Note: at
# 1,400 kg/ha and 60% share, total lint = 227 kt and domestic lint =
# 51 kt × 1.20 = 62 kt, leaving 165 kt for export (Scenarios sheet I13).
# ─────────────────────────────────────────────────────────────────────────────
# MANUFACTURING_HIGH_YIELD = Scenario(
#     name="manufacturing_high_yield",
#     description=(
#         "Combined: yield 1,400 kg/ha and 60% domestic textile processing. "
#         "Maximises on-farm output and downstream employment."
#     ),
#     yield_kg_ha=YIELD_TARGET_KG_HA,
#     organic_share=0.40,
#     land_occupation_m2=round(10_000 / YIELD_TARGET_KG_HA, 6),
#     fertilizer_n_kg_manure=0.017864,                 # same per-kg as HIGH_YIELD / BASELINE
#     fertilizer_n_kg_urea=0.006627,
#     fertilizer_p_kg=0.008900,
#     pesticide_kg=0.00052899 + 2.7842e-5,
#     profenofos_kg=0.00052899,
#     lambda_cyhalothrin_kg=2.7842e-5,
#     n_emission_kg=0.024491,
#     p_emission_kg=P2O5_TO_P * 0.008900,   # 0.008900 kg P2O5 -> 0.003884 kg P
#     clothing=ClothingParams(
#         domestic_fabric_kg_yr=3 * 17_159_408.3,
#     ),
#     farm_income=FarmIncomeParams(
#         farmgate_price_usd_per_kg_sc=0.4784,
#         input_cost_usd_per_ha=INPUT_COST_CONV_USD_KG * YIELD_TARGET_KG_HA,
#     ),
#     domestic_textile_t_override=3 * 17_159.408311,
#     employment=EmploymentParams(
#         farm_person_days_per_ha=265.114,
#     ),
# )


# ─────────────────────────────────────────────────────────────────────────────
# 7. IRRIGATION — full supplemental irrigation, India national average yield
# ─────────────────────────────────────────────────────────────────────────────
# Yield: 1,430.24 kg/ha = India national average irrigated cotton yield
#   (~2.26× Tanzania baseline of 633.69 kg/ha; conservative relative to
#   Kansas field trial at 4.7×; consistent with CICR +125.7% benchmark).
#   Source: WFN Report 68 (Chapagain et al.) / Scenarios sheet J2.
# Per-ha inputs: FIXED at BASELINE levels — irrigation increases yield, not
#   agrochemical application. Same kg/ha of fertilizer, pesticide, and labour
#   as BASELINE → per-kg values scale DOWN by BL_YIELD/IRR_YIELD = 0.4431.
#     fertilizer_n_kg_manure = 0.017864 × 0.4431 = 0.007915 kg/kg
#     fertilizer_n_kg_urea   = 0.006627 × 0.4431 = 0.002936 kg/kg
#     fertilizer_p_kg        = 0.008900 × 0.4431 = 0.003943 kg/kg
#     profenofos_kg          = 0.00052899 × 0.4431 = 2.344e-4 kg/kg
#     lambda_cyhalothrin_kg  = 2.7842e-5 × 0.4431 = 1.234e-5 kg/kg
#     n_emission_kg          = 0.024491 × 0.4431 = 0.010851 kg/kg
#     p_emission_kg          = 0.008900 × 0.4431 = 0.003943 kg/kg
# Irrigation water: 3.402139 m³/kg raw cotton (new input added on top)
#   Source: India national average blue water footprint (Scenarios sheet J14;
#   WFN Report 68). Maps to Water,river biosphere flow.
# Labour: scales fully with yield, matching HIGH_YIELD's convention:
#   120 x (1430.24/633.69) = 270.84 person-days/ha (Scenarios sheet).
# input_cost_usd_per_ha: held at BASELINE level (56.0 USD/ha); irrigation
#   raises yield from the SAME agrochemical expenditure.
# ─────────────────────────────────────────────────────────────────────────────
_IRR_SCALE = YIELD_BASELINE_KG_HA / YIELD_IRRIGATION_KG_HA   # = 0.44307

IRRIGATION = Scenario(
    name="irrigation",
    description=(
        "Full supplemental irrigation raises yield to 1,430 kg/ha (India "
        "national average irrigated cotton yield; ~2.26× Tanzania baseline). "
        "Per-ha fertilizer and pesticide FIXED at baseline levels — "
        "irrigation is the sole driver of yield gain, so per-kg agrochemical "
        "burdens fall by 56%. Labour scales with yield (270.84 person-days/ha). "
        "Irrigation water 3.402 m³/kg (WFN Report 68)."
    ),
    yield_kg_ha=YIELD_IRRIGATION_KG_HA,                 # 1430.237547 (Scenarios sheet J2)
    organic_share=0.40,
    land_occupation_m2=10_000/YIELD_IRRIGATION_KG_HA,   # = 6.993 m²/kg
    # seed_replant_kg is deliberately LEFT at the baseline per-kg value (0.0885),
    # matching HIGH_YIELD, so the seed-replant technosphere loop multiplier is
    # unchanged and total land use stays at baseline (irrigation reads as a
    # water-focused perturbation, not a land change). Holding seed per-ha
    # instead would shrink the loop and cut land use ~5%.
    fertilizer_n_kg_manure=0.017864 * _IRR_SCALE,       # fixed per-ha → 0.007915 kg/kg
    fertilizer_n_kg_urea=0.006627 * _IRR_SCALE,         # fixed per-ha → 0.002936 kg/kg
    fertilizer_p_kg=0.008900 * _IRR_SCALE,              # fixed per-ha → 0.003943 kg/kg
    pesticide_kg=(0.00052899 + 2.7842e-5) * _IRR_SCALE,
    profenofos_kg=0.00052899 * _IRR_SCALE,              # fixed per-ha → 2.344e-4 kg/kg
    lambda_cyhalothrin_kg=2.7842e-5 * _IRR_SCALE,       # fixed per-ha → 1.234e-5 kg/kg
    n_emission_kg=0.024491 * _IRR_SCALE,                # fixed per-ha → 0.010851 kg/kg
    p_emission_kg=P2O5_TO_P * 0.008900 * _IRR_SCALE,    # fixed per-ha; P2O5 0.003943 -> P 0.001721
    # Tractor passes (ploughing, spraying, weeding) are per-HECTARE field
    # operations, so likewise fixed per-ha. NOTE: harvest/haulage arguably
    # scales with output, but the model lumps all tractor use into one field;
    # scaling it fully is the choice consistent with fertilizer/pesticide.
    tractor_kg=1.4159e-06 * _IRR_SCALE,                 # fixed per-ha → 6.2733e-07
    irrigation_water_m3=3.402139,                       # Scenarios sheet J14; WFN Report 68
    farm_income=FarmIncomeParams(
        farmgate_price_usd_per_kg_sc=0.4784,            # Scenarios sheet J15
        input_cost_usd_per_ha=INPUT_COST_CONV_USD_KG * YIELD_BASELINE_KG_HA,  # 56.0 USD/ha (fixed)
    ),
    employment=EmploymentParams(
        farm_person_days_per_ha=270.84,                 # 120 x (1430.24/633.69); scales with
                                                        # yield, same convention as HIGH_YIELD
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXTENSIFICATION — baseline per-hectare intensity spread over a larger
#    area, delivering the SAME national output as HIGH_YIELD.
# ─────────────────────────────────────────────────────────────────────────────
# This is the area-expansion counterfactual to HIGH_YIELD. Both
# scenarios deliver identical national output; they differ only in HOW:
#
#   HIGH_YIELD       : 2,471 kg/ha on   445,817 ha -> intensive, same land
#   EXTENSIFICATION :   634 kg/ha on 1,738,417 ha -> extensive, 3.90x land
#
# Every per-kg field rate stays at the BASELINE default, because both intensity
# and area scale together: baseline per-ha rate x 3.90 area / 3.90 output =
# baseline per-kg. Only the land requirement changes.
#
# INTERPRETATION CAVEAT: with a single land-use characterisation factor shared
# by all scenarios, this scenario is by construction identical to HIGH_YIELD in
# every category EXCEPT land use, where it is 3.90x worse. Turning it into a
# genuine sparing/sharing test would additionally require intensity-
# differentiated land CFs (Cropland_Minimal vs Cropland_Intense) and a land
# TRANSFORMATION term for the ~1.29 million ha of new cropland. Both are
# deliberately out of scope here; see the Discussion.
# ─────────────────────────────────────────────────────────────────────────────
_EXT_AREA_FACTOR = YIELD_TARGET_KG_HA / YIELD_BASELINE_KG_HA   # = 3.8994

EXTENSIFICATION = Scenario(
    name="extensification",
    description=(
        "Area-expansion counterfactual to HIGH_YIELD: national output matched "
        "to the TCDS target (~1.10 Mt) by expanding cultivated area 3.90x at "
        "unchanged baseline yield (633.69 kg/ha) and unchanged per-hectare "
        "input intensity. Per-kg field rates therefore equal baseline; only "
        "land requirement differs."
    ),
    yield_kg_ha=YIELD_BASELINE_KG_HA,                  # baseline intensity
    national=NationalContext(
        national_area_ha=BASELINE_AREA_HA * _EXT_AREA_FACTOR   # = 1,738,417 ha
    ),
    organic_share=0.40,
    land_occupation_m2=10_000/YIELD_BASELINE_KG_HA,    # = 15.7806 m2/kg (baseline)
    # All remaining per-kg field rates intentionally left at the BASELINE
    # dataclass defaults (fertiliser, pesticide, N/P emissions, tractor, seed).
    farm_income=FarmIncomeParams(
        farmgate_price_usd_per_kg_sc=0.4784,
        input_cost_usd_per_ha=INPUT_COST_CONV_USD_KG * YIELD_BASELINE_KG_HA,  # 56.0
    ),
    employment=EmploymentParams(
        farm_person_days_per_ha=120.0,                 # baseline per-ha labour
    ),
)


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO REGISTRY (for iteration in notebooks)
# ═════════════════════════════════════════════════════════════════════════════

ALL_SCENARIOS = [
    BASELINE,
    HIGH_YIELD,
    EXTENSIFICATION,
    ORGANIC_EXPANSION,
    MANUFACTURING_EXPANSION,
    IRRIGATION,
]


if __name__ == "__main__":
    print(f"{'scenario':24s} {'yield':>7s}  {'organic':>8s}  "
          f"{'N/kg':>8s}  {'P/kg':>8s}  {'pest/kg':>10s}  "
          f"{'price USD/kg':>12s}  {'prod (kt)':>10s}")
    for s in ALL_SCENARIOS:
        print(f"{s.name:24s} {s.yield_kg_ha:7.1f}  "
              f"{s.organic_share*100:7.0f}%  "
              f"{s.fertilizer_n_kg_manure + s.fertilizer_n_kg_urea:8.4f}  {s.fertilizer_p_kg:8.4f}  "
              f"{s.pesticide_kg:10.6f}  "
              f"{s.farm_income.farmgate_price_usd_per_kg_sc:12.4f}  "
              f"{s.national_production_t()/1000:10.1f}")
