"""
needs.py — Decent Living Standards (DLS) coverage indicators for the
Tanzania cotton LCA.

Translates cotton industry physical/monetary flows into "person-years of
need covered," following the logic of Rao & Min (2018) DLS + Cassidy et
al. (2013) people-nourished-per-hectare + Neugebauer et al. (2017) fair
wage gap.

DLS components and cotton-industry pathways
-------------------------------------------
    Component      Pathway                              Implemented
    ─────────────  ───────────────────────────────────  ────────────
    Clothing       lint  → garments                     yes (direct)
    Nutrition      seed  → edible oil + dairy feed      yes (direct)
    Income         farmgate + wage income vs. Anker LI  yes (monetary)
    Decent work    jobs supported per tonne / per ha    yes (jobs)
    Shelter        —                                    no pathway
    Water/sanit.   —                                    no pathway*
    Health         —                                    no pathway
    Education      —                                    no pathway
    Mobility       —                                    no pathway
    ICT            —                                    no pathway

    *Water actually shows up on the ENVIRONMENTAL CEILING side
     (consumption/pollution) — see the doughnut/SJOS framing, not here.

Units convention
----------------
All functions return dictionaries keyed on explicit names; primary
output is "person-years of need covered per hectare" (or per tonne,
where specified). Multiply by district hectares (from SPAM weights)
and divide by district population for a local coverage ratio.

References (verify before citing)
---------------------------------
Rao, N. D. & Min, J. (2018) Soc. Indic. Res. 138.
Millward-Hopkins et al. (2020) Glob. Env. Change 65.
Cassidy, E. S. et al. (2013) Environ. Res. Lett. 8, 034015.
Neugebauer, S. et al. (2017) J. Cleaner Prod. 143.
Anker, R. & Anker, M. (2017) Living Wages Around the World. Elgar.
USDA FoodData Central (cottonseed oil energy density).
FAO Feedipedia (cottonseed meal protein).
NRC (2001) Nutrient Requirements of Dairy Cattle.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ═════════════════════════════════════════════════════════════════════════════
# 1. NUTRITION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class NutritionParams:
    """Parameters for seed-co-product nutritional delivery."""
    # Ginning outturn (per kg seed cotton)
    # Source: cotton_industry_data.xlsx Parameters sheet (TCB)
    #   c = 0.38 (lint weight ratio of raw cotton)
    #   d = 0.04 (ginning loss ratio)
    #   f_seed = 1 - c - d by mass balance
    f_lint:  float = 0.38
    f_seed:  float = 0.58
    f_trash: float = 0.04

    # Oil-mill split (per kg cottonseed in)
    f_oil_refined: float = 0.135   # USDA; crude 0.16 × ~85% refining yield
    f_meal:        float = 0.45    # FAO Feedipedia
    f_hulls:       float = 0.25
    f_linters:     float = 0.09

    # Nutritional content
    kcal_oil:           float = 8_840   # kcal / kg refined oil (USDA FDC)
    kcal_meal_ME:       float = 2_700   # kcal metabolizable energy / kg (NRC)
    protein_meal_kg_kg: float = 0.410   # kg crude protein / kg meal

    # Animal conversion efficiencies (Cassidy 2013 Table S1)
    eta_kcal_milk:    float = 0.24
    eta_protein_milk: float = 0.28
    eta_kcal_beef:    float = 0.03
    eta_protein_beef: float = 0.04

    # DLS thresholds (Rao & Min 2018)
    kcal_per_cap_yr:      float = 766_500   # 2,100 kcal/day
    protein_per_cap_yr_kg: float = 18.25    # 50 g/day


def nutrition_per_ha(
    yield_kg_ha: float,
    seed_replant_kg_per_kg_rc: float = 0.0885,
    p: NutritionParams | None = None,
    animal: str = "milk",
) -> dict:
    """Person-years of kcal and protein need covered per hectare.

    Parameters
    ----------
    yield_kg_ha : yield in kg seed cotton per ha
    seed_replant_kg_per_kg_rc : fraction of seed retained for replanting
    animal : 'milk' or 'beef' — fate of cottonseed meal in ruminant system
    """
    p = p or NutritionParams()
    eta_k = getattr(p, f"eta_kcal_{animal}")
    eta_p = getattr(p, f"eta_protein_{animal}")

    seed_to_mill_frac = p.f_seed - seed_replant_kg_per_kg_rc
    seed_kg_ha = yield_kg_ha * seed_to_mill_frac

    kcal_oil  = seed_kg_ha * p.f_oil_refined * p.kcal_oil
    kcal_meal = seed_kg_ha * p.f_meal * p.kcal_meal_ME * eta_k
    protein_kg = seed_kg_ha * p.f_meal * p.protein_meal_kg_kg * eta_p

    return {
        "kcal_oil_ha":       kcal_oil,
        "kcal_meal_ha":      kcal_meal,
        "kcal_total_ha":     kcal_oil + kcal_meal,
        "protein_kg_ha":     protein_kg,
        "ppl_yr_kcal_ha":    (kcal_oil + kcal_meal) / p.kcal_per_cap_yr,
        "ppl_yr_protein_ha": protein_kg / p.protein_per_cap_yr_kg,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. CLOTHING
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ClothingParams:
    """Parameters for clothing-need coverage.

    Domestic fabric output is set directly per scenario — no yield- or
    lint-based derivation. The LCA system already accounts for ginning
    losses and textile conversion losses inside the textile activity,
    so DLS clothing coverage just needs the absolute kg of fabric the
    Tanzanian textile sector produces in a year.
    """
    # Absolute domestic fabric (textile) output, kg/yr.
    # Baseline: 17.159 kt/yr, matching Scenario._baseline_textile_t exactly.
    # (This previously read 16,929,975 kg, 1.34% below the textile output the
    # LCA and the employment indicator both use, which understated baseline
    # clothing coverage and made the manufacturing-expansion clothing
    # fold-change read 3.04x instead of 3.00x.)
    # Manufacturing-expansion scenarios triple this to 51.478 kt/yr.
    domestic_fabric_kg_yr: float = 17_159_408.3

    # Garment-equivalent factors (kg fabric per garment) — informational
    kg_fabric_per_shirt:   float = 0.22  # basic T-shirt
    kg_fabric_per_trouser: float = 0.55  # basic trousers
    kg_fabric_per_bedding: float = 1.0   # sheet set

    # DLS clothing threshold — CONTESTED, see notes
    kg_fabric_per_cap_yr: float = 3.0
    # Millward-Hopkins et al. 2020 use ~3 kg/cap/yr new clothing mass.


def clothing_per_ha(
    yield_kg_ha: float,
    p: ClothingParams | None = None,
    cotton_share_of_clothing: float = 1.0,
    national_area_ha: float = 445_817.0,
) -> dict:
    """Per-hectare equivalents of clothing-need coverage.

    Domestic fabric output is taken directly from `p.domestic_fabric_kg_yr`
    (set per scenario). The per-ha figures are just the absolute output
    divided by national harvested area, retained so downstream code that
    multiplies by `national_area_ha` recovers the absolute total.

    `cotton_share_of_clothing` scales the threshold by the fraction of
    the DLS clothing floor realistically met by cotton.
    `yield_kg_ha` is unused in the fabric path; kept for signature
    compatibility with the other `*_per_ha` functions.
    """
    p = p or ClothingParams()
    fabric_kg_ha  = p.domestic_fabric_kg_yr / national_area_ha
    threshold = p.kg_fabric_per_cap_yr * cotton_share_of_clothing
    return {
        "fabric_kg_yr":       p.domestic_fabric_kg_yr,
        "fabric_kg_ha":       fabric_kg_ha,
        "shirt_equiv_ha":     fabric_kg_ha / p.kg_fabric_per_shirt,
        "trouser_equiv_ha":   fabric_kg_ha / p.kg_fabric_per_trouser,
        "ppl_yr_clothing_ha": fabric_kg_ha / threshold,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3a. FARM INCOME (living-income coverage, household basis)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FarmIncomeParams:
    """Parameters for smallholder farmer income vs. Anker living-income.

    Monetary values in USD. Convert TSH externally (~2,500 TSH/USD,
    2023–24).
    """
    # Farmgate economics (TZ smallholder cotton, 2022–23 indicative)
    farmgate_price_usd_per_kg_sc: float = 0.46   # mirrors DLS_Parameters sheet
    input_cost_usd_per_ha:         float = 56.0   # seed + pesticide + hired labor;
                                                   # = 0.088371 USD/kg (Scenarios sheet,
                                                   # interviews) x 633.69 kg/ha. Bwana et
                                                   # al. (2020) Table 4 cash items (~60)
                                                   # corroborate.
    avg_farm_size_ha:              float = 1.2    # TZ smallholder cotton average
    household_size:                float = 5.0

    # Anker living-income benchmark (household/yr) — Tanzanian rural
    living_income_usd_per_hh_yr: float = 2_508.0   # mirrors DLS_Parameters sheet

    # Cotton's share of total household cash income
    # (smallholders grow maize, sorghum etc. — cotton typically
    # 40–70% of cash income per TCB/CmiA surveys)
    cotton_share_of_hh_income: float = 0.60


def farm_income_per_ha(
    yield_kg_ha: float,
    p: FarmIncomeParams | None = None,
) -> dict:
    """Farmer net income and living-income coverage, per hectare
    cultivated and per household."""
    p = p or FarmIncomeParams()
    gross = yield_kg_ha * p.farmgate_price_usd_per_kg_sc
    net   = gross - p.input_cost_usd_per_ha
    hh_income_from_cotton   = net * p.avg_farm_size_ha
    implied_total_hh_income = hh_income_from_cotton / p.cotton_share_of_hh_income

    return {
        "gross_usd_ha":           gross,
        "net_usd_ha":             net,
        "cotton_income_usd_hh":   hh_income_from_cotton,
        "implied_total_usd_hh":   implied_total_hh_income,
        "living_income_coverage": implied_total_hh_income / p.living_income_usd_per_hh_yr,
        "living_income_gap_hh":   max(0.0,
                                      p.living_income_usd_per_hh_yr
                                      - implied_total_hh_income),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3b. WAGE-WORKER INCOME (ginning, textile, oil milling)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class WageParams:
    """Annual wages and living-wage benchmarks for downstream workers.

    All PLACEHOLDERS — verify against TZ minimum-wage schedules
    (LITA 2022 revision), 21st Century Textiles payroll, and Anker
    living-wage reports for TZ urban/semi-urban manufacturing.
    """
    # Annual take-home wages (USD/yr per worker)
    ginning_wage_usd_yr:  float = 1500   # ~125 USD/mo, verify
    textile_wage_usd_yr:  float = 1500   # similar to ginning, but verify
    oilmill_wage_usd_yr:  float = 1500   # similar to ginning

    # Anker living-wage benchmark (per WORKER, not household)
    # ~$200/mo is an adjacent reference point.
    living_wage_usd_yr: float = 2_268   # mirrors DLS_Parameters sheet


def wage_income_per_ha(
    employment: dict,
    p: WageParams | None = None,
) -> dict:
    """Wage bill and living-wage coverage for downstream workers,
    expressed per hectare of upstream cotton cultivation.

    Consumes the dict returned by `employment_per_ha` so stage FTEs
    stay internally consistent.
    """
    p = p or WageParams()

    # Oil milling carries no FTE estimate (see EmploymentParams), so it has no
    # wage bill here. Its per-worker living-wage coverage below is a ratio of
    # wage rates and does not depend on headcount, so that figure survives.
    wages_ha = {
        "ginning": employment["ginning_fte_ha"] * p.ginning_wage_usd_yr,
        "textile": employment["textile_fte_ha"] * p.textile_wage_usd_yr,
    }

    coverage = {
        "ginning": p.ginning_wage_usd_yr / p.living_wage_usd_yr,
        "textile": p.textile_wage_usd_yr / p.living_wage_usd_yr,
        "oilmill": p.oilmill_wage_usd_yr / p.living_wage_usd_yr,
    }
    gap = {
        stage: max(0.0, p.living_wage_usd_yr - wage)
        for stage, wage in [
            ("ginning", p.ginning_wage_usd_yr),
            ("textile", p.textile_wage_usd_yr),
            ("oilmill", p.oilmill_wage_usd_yr),
        ]
    }

    return {
        "ginning_wages_usd_ha":         wages_ha["ginning"],
        "textile_wages_usd_ha":         wages_ha["textile"],
        "total_wages_usd_ha":           sum(wages_ha.values()),
        "ginning_lw_coverage":          coverage["ginning"],
        "textile_lw_coverage":          coverage["textile"],
        "oilmill_lw_coverage":          coverage["oilmill"],
        "ginning_lw_gap_usd_yr_worker": gap["ginning"],
        "textile_lw_gap_usd_yr_worker": gap["textile"],
        "oilmill_lw_gap_usd_yr_worker": gap["oilmill"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. DECENT WORK (jobs supported)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EmploymentParams:
    """Jobs-per-activity coefficients along the cotton chain."""
    # Farming: person-days / ha / season
    farm_person_days_per_ha: float = 120.0
    working_days_per_fte:    float = 250.0

    # Ginning: FTE / tonne lint processed
    # Derived from: China-Africa Cotton 3,300 workers (~20% share)
    # + Alliance Ginneries 290 FTE (TZ+ZM), annualised for ~4-month season.
    # Range 0.04–0.07 FTE/t lint; midpoint 0.06.
    ginning_fte_per_t_lint: float = 0.06

    # Textile manufacturing: FTE / tonne textile
    # 21st Century Textiles: 2,200 FTE / (20 t fabric/day × 330 days) = 0.33
    textile_fte_per_t_fabric: float = 0.33

    # Oil milling contributes NO employment term. The FTE/tonne-seed
    # coefficient it would require has no located source for Tanzanian mills,
    # and carrying a placeholder shifted employment coverage by 3.3-7.3% --
    # enough to reorder irrigation and manufacturing_expansion -- on no
    # evidence. Oil milling remains represented in the DLS assessment through
    # the nutrition pathway, whose crushing yields are literature values
    # (USDA refined-oil yield, FAO Feedipedia meal yield).

    # Indirect jobs (MRIO-style multiplier over direct) - Exiobase agriculture multiplier ≈ 0.3 for low-income
    indirect_multiplier: float = 0.3

    # Fraction of national lint output processed domestically into textile.
    # NOT a primary scenario control any more — `assess_needs_from_scenario`
    # overwrites this with a value derived from the scenario's absolute
    # `domestic_textile_t_override` lever (so the share is observed, not
    # set). The default is retained for standalone uses of
    # `employment_per_ha` outside the scenario flow.
    domestic_processing_share: float = 1.0


def employment_per_ha(
    yield_kg_ha: float,
    p: EmploymentParams | None = None,
    mass_fractions: NutritionParams | None = None,
) -> dict:
    """FTE-equivalent jobs supported per hectare, summed across farming,
    ginning and textile manufacturing.

    Oil milling is deliberately excluded; see EmploymentParams for why.
    """
    p  = p  or EmploymentParams()
    mf = mass_fractions or NutritionParams()

    # Farm
    farm_fte = p.farm_person_days_per_ha / p.working_days_per_fte

    # Downstream (tonnes per ha)
    lint_t = yield_kg_ha * mf.f_lint / 1_000
    # Only the lint share that is processed domestically generates textile-mill FTEs.
    fabric_t = lint_t * p.domestic_processing_share / 1.201

    ginning_fte = lint_t * p.ginning_fte_per_t_lint
    textile_fte = fabric_t * p.textile_fte_per_t_fabric

    direct   = farm_fte + ginning_fte + textile_fte
    indirect = direct * p.indirect_multiplier

    return {
        "farm_fte_ha":     farm_fte,
        "ginning_fte_ha":  ginning_fte,
        "textile_fte_ha":  textile_fte,
        "direct_fte_ha":   direct,
        "indirect_fte_ha": indirect,
        "total_fte_ha":    direct + indirect,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. ROLL-UP
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class NeedsAssessment:
    yield_kg_ha:  float
    nutrition:    dict
    clothing:     dict
    farm_income:  dict
    wage_income:  dict
    employment:   dict

    def summary(self) -> dict:
        """Flatten headline per-ha numbers into one dict for tabulation."""
        return {
            "yield_kg_ha":         self.yield_kg_ha,
            "ppl_yr_kcal_ha":      self.nutrition["ppl_yr_kcal_ha"],
            "ppl_yr_protein_ha":   self.nutrition["ppl_yr_protein_ha"],
            "ppl_yr_clothing_ha":  self.clothing["ppl_yr_clothing_ha"],
            "farm_li_coverage":    self.farm_income["living_income_coverage"],
            "ginning_lw_coverage": self.wage_income["ginning_lw_coverage"],
            "textile_lw_coverage": self.wage_income["textile_lw_coverage"],
            "oilmill_lw_coverage": self.wage_income["oilmill_lw_coverage"],
            "total_fte_ha":        self.employment["total_fte_ha"],
            "total_wages_usd_ha":  self.wage_income["total_wages_usd_ha"],
        }


def assess_needs(
    yield_kg_ha: float = 400.0,
    seed_replant_kg_per_kg_rc: float = 0.0885,
    animal: str = "milk",
    cotton_share_of_clothing: float = 0.5,
) -> NeedsAssessment:
    """Run all indicators with default parameters.

    Example
    -------
    >>> a = assess_needs(yield_kg_ha=400)
    >>> a.summary()
    """
    empl = employment_per_ha(yield_kg_ha)
    return NeedsAssessment(
        yield_kg_ha=yield_kg_ha,
        nutrition=nutrition_per_ha(yield_kg_ha, seed_replant_kg_per_kg_rc,
                                   animal=animal),
        clothing=clothing_per_ha(yield_kg_ha,
                                 cotton_share_of_clothing=cotton_share_of_clothing),
        farm_income=farm_income_per_ha(yield_kg_ha),
        wage_income=wage_income_per_ha(empl),
        employment=empl,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 6. NATIONAL ROLL-UP
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class NationalContext:
    """National scaling parameters — VERIFY all for reference year."""
    national_area_ha:    float = 445_817      # = 282,510 t / 633.69 kg/ha; consistent
                                              # with Parameters 'Total planted area'
                                              # (706,000 ha was the 400-kg/ha-era pair)
                                               # (= 282,509.62 t ÷ 400 kg/ha, TCB 2024)
    population:          float = 67_400_000   # UN WPP 2024 medium — verify
    rural_labor_force:   float = 14_100_000   # Only includes LF in cotton growing regions
    kcal_need_total_yr:  float = None         # filled from NutritionParams

    def __post_init__(self):
        if self.kcal_need_total_yr is None:
            self.kcal_need_total_yr = (
                self.population * NutritionParams().kcal_per_cap_yr
            )


def national_assessment(
    a: NeedsAssessment,
    ctx: NationalContext | None = None,
) -> dict:
    """Scale a per-ha NeedsAssessment to national totals and express
    each indicator as a fraction of the corresponding national need.
    """
    ctx = ctx or NationalContext()
    A = ctx.national_area_ha

    # Absolute national totals
    kcal_nat     = a.nutrition["kcal_total_ha"]   * A
    protein_nat  = a.nutrition["protein_kg_ha"]   * A
    fabric_nat   = a.clothing["fabric_kg_ha"]     * A
    fte_nat      = a.employment["total_fte_ha"]   * A
    wages_nat    = a.wage_income["total_wages_usd_ha"] * A
    farm_net_nat = a.farm_income["net_usd_ha"]    * A

    # National-need coverage fractions
    frac_kcal     = kcal_nat    / (ctx.population * NutritionParams().kcal_per_cap_yr)
    frac_protein  = protein_nat / (ctx.population * NutritionParams().protein_per_cap_yr_kg)
    frac_clothing = fabric_nat  / (ctx.population * ClothingParams().kg_fabric_per_cap_yr)
    frac_jobs     = fte_nat     / ctx.rural_labor_force

    return {
        # absolute
        "national_kcal_yr":        kcal_nat,
        "national_protein_kg_yr":  protein_nat,
        "national_fabric_kg_yr":   fabric_nat,
        "national_fte":            fte_nat,
        "national_wages_usd_yr":   wages_nat,
        "national_farm_net_usd":   farm_net_nat,
        # coverage of national DLS floors
        "frac_nat_kcal_need":      frac_kcal,
        "frac_nat_protein_need":   frac_protein,
        "frac_nat_clothing_need":  frac_clothing,
        "frac_rural_labor_force":  frac_jobs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 7. SCENARIO-DRIVEN ASSESSMENT
# ═════════════════════════════════════════════════════════════════════════════

def assess_needs_from_scenario(scenario) -> NeedsAssessment:
    """Run the full DLS assessment using a Scenario's parameters.

    Reads yield, cotton_share_of_clothing, and all six DLS parameter
    dataclasses directly off the scenario — no shared defaults, so a
    scenario fully determines its DLS output.

    The textile-mill processing share used in employment is DERIVED from
    the scenario's absolute `domestic_textile_t_override` lever (via
    `Scenario.effective_processing_share()`), not taken from
    `EmploymentParams.domestic_processing_share`. This keeps total
    domestic textile output as the single textile-industry control.

    Accepts a `scenarios.Scenario` (imported lazily to avoid a circular
    import at module load time).
    """
    from dataclasses import replace

    yield_kg_ha = scenario.yield_kg_ha

    # Derive the processing share from the absolute textile lever, then
    # overwrite it on a copy of the scenario's EmploymentParams so the
    # per-ha textile FTE calculation in employment_per_ha sees the right
    # value.
    eff_share = scenario.effective_processing_share()
    empl_params = replace(scenario.employment, domestic_processing_share=eff_share)

    empl = employment_per_ha(
        yield_kg_ha,
        p=empl_params,
        mass_fractions=scenario.nutrition,
    )
    return NeedsAssessment(
        yield_kg_ha=yield_kg_ha,
        nutrition=nutrition_per_ha(
            yield_kg_ha,
            seed_replant_kg_per_kg_rc=scenario.seed_replant_kg,
            p=scenario.nutrition,
        ),
        clothing=clothing_per_ha(
            yield_kg_ha,
            p=scenario.clothing,
            cotton_share_of_clothing=scenario.cotton_share_of_clothing,
            national_area_ha=scenario.national.national_area_ha,
        ),
        farm_income=farm_income_per_ha(yield_kg_ha, p=scenario.farm_income),
        wage_income=wage_income_per_ha(empl, p=scenario.wage),
        employment=empl,
    )


if __name__ == "__main__":
    a   = assess_needs(yield_kg_ha=400.0)
    nat = national_assessment(a)

    print("── Per-ha ──")
    for k, v in a.summary().items():
        print(f"  {k:22s} {v:14.4f}")

    print("\n── National ──")
    for k, v in nat.items():
        print(f"  {k:26s} {v:16.2f}")
