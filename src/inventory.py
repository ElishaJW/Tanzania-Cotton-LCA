"""
inventory.py — Foreground database builder for the Tanzania Cotton LCA.

The single public function `build_foreground_db(scenario, spatial_data)` wipes and
rebuilds the 'foreground' Brightway database for a given scenario. All foreground
nodes (product nodes, process stubs, district activities, national aggregators)
and their edges are created from scratch on each call.

Spatial data (production weights, ginning/textile weights, district shapefile) must be
precomputed before the scenario loop and passed in via the `spatial_data` dict —
these are derived from fixed input files and do not change between scenarios.

Parameters of `spatial_data`
-----------------------------
    'tz_districts_gdf'   : GeoDataFrame — Tanzania ADM2 districts
    'production_weights' : dict[str, float] — {ADM2_PCODE: fraction} summing to 1.0
    'ginning_weights'    : dict[str, float] — {ADM2_PCODE: fraction} summing to 1.0
    'textile_weights'    : dict[str, float] — {ADM2_PCODE: fraction} summing to 1.0
"""

import math

import bw2data as bd
from src.config import (
    DB_BIOSPHERE, DB_ECOINVENT, DB_FOREGROUND,
    LINT_ALLOC, SEED_ALLOC, OIL_ALLOC, CAKE_ALLOC,
)


# ── Exchange-level uncertainty (stats_arrays format) ────────────────────────
# All foreground distributions are LOGNORMAL (stats_arrays type 2). Rationale:
# strictly-positive flows, closed under products/reciprocals, matches the
# multiplicative pedigree method. σ_ln values are the "sigma_ln TOTAL" column of
# the "Pedigree" sheet in cotton_industry_data.xlsx (measured spread combined in
# quadrature with the Weidema/Ciroth pedigree data-quality penalty).
#
# Keyed by scenarios.py parameter name. A parameter absent here carries no
# uncertainty (deliberately: textile_lint_kg, textile_waste_kg, ginning_waste_kg,
# seed_replant_kg — see the Uncertainty sheet).
_SIGMA_LN = {
    "land_occupation_m2":      0.502,
    "pesticide_kg":            0.541,   # also applied to profenofos_kg / lambda_cyhalothrin_kg
    "n_emission_kg":           0.843,
    "p_emission_kg":           0.843,
    "fertilizer_n_kg_manure":  0.437,
    "fertilizer_n_kg_urea":    0.437,
    "fertilizer_p_kg":         0.437,
    "tractor_kg":              0.513,
    "ginning_transport_tkm":   0.442,
    "ginning_electricity_kwh": 0.442,
    "textile_coal_kwh":        0.455,
    "textile_water_river_m3":  0.464,
    "textile_water_air_m3":    0.464,
    "textile_electricity_kwh": 0.441,
    "oil_electricity_kwh":     0.745,
}


def _ln_unc(amount, param):
    """stats_arrays lognormal fields for a positive `amount` with the σ_ln of
    `param`. loc = ln(amount); lognormal is scale-invariant, so any constant
    multiplier applied to `amount` before the edge (allocation share, unit
    conversion, EF) shifts loc without changing σ. Returns {} for a missing
    param or non-positive amount so the edge falls back to deterministic."""
    sln = _SIGMA_LN.get(param)
    if sln is None or amount is None or amount <= 0:
        return {}
    return dict(uncertainty_type=2, loc=math.log(amount), scale=sln)

# OLCA-Pest emission fractions for Oil-Bearing crops × Insecticide × No-buffer
# (Speck et al. 2022, Int J LCA, supplementary table SC38). Cotton classed as
# oil-bearing (cottonseed produces a major vegetable oil); insecticide class
# matches the bollworm-driven pest pressure of TZ cotton; no-buffer reflects
# smallholder fields without engineered vegetative strips. Crop-uptake
# fraction (~0.799) is NOT an environmental emission and is therefore omitted.
_OLCA_F_AIR        = 0.10        # → air, low population density
_OLCA_F_SOIL_AGRI  = 0.094       # → soil, agricultural
_OLCA_F_SOIL_NAT   = 0.0073      # → soil, forestry (off-field natural soil)
_OLCA_F_WATER_SURF = 0.000251    # → water, surface water (off-field runoff)

# Field GHG emission factors from N fertilizer application (IPCC 2019 Vol 4
# Ch 11). Tanzania's Lake-Zone cotton belt qualifies as "dry climate" under
# IPCC categorisation, so the dry-climate EFs are used.
_EF_N2O_DIRECT_SYNTH = 0.010     # kg N₂O-N per kg synthetic N applied (incl. urea)
_EF_N2O_DIRECT_ORG   = 0.006     # kg N₂O-N per kg organic N applied (manure)
_EF_N2O_INDIRECT     = 0.0014    # kg N₂O-N per kg N applied (volatilisation + leaching combined)
_N2O_PER_N           = 44.0/28.0 # mass conversion N₂O-N → N₂O (≈1.5714)
_UREA_N_CONTENT      = 0.46      # kg N per kg urea (stoichiometry CO(NH₂)₂)
_UREA_CO2_EF         = 0.733     # kg CO₂ per kg urea applied (IPCC: 0.20 kg C × 44/12)
# textile_lint_kg and textile_waste_kg deliberately carry NO uncertainty: the
# documented ± is daily throughput scatter at one site, not uncertainty in the
# annual-basis mass ratio. Coal / water uncertainty now comes from _SIGMA_LN.


# ── Biosphere node lookup ────────────────────────────────────────────────────

def _get_bio_node(db, name, categories):
    """Return a biosphere node by name + compartment; falls back to name-only match."""
    cats = tuple(categories)
    matches = [a for a in db
               if a["name"] == name
               and tuple(a.get("categories", ())) == cats]
    if not matches:
        matches = [a for a in db if a["name"] == name]
    if matches:
        return matches[0]
    raise ValueError(f"Biosphere node not found: '{name}' {categories}")


def _get_bio_node_first(db, name, compartment_options):
    """Return the first biosphere node found across a list of candidate compartments."""
    for cats in compartment_options:
        ct = tuple(cats)
        hits = [a for a in db
                if a["name"] == name
                and tuple(a.get("categories", ())) == ct]
        if hits:
            return hits[0]
    raise ValueError(f"Biosphere node not found: {name!r} in any of {compartment_options}")


# ── Main builder ─────────────────────────────────────────────────────────────

def build_foreground_db(scenario, spatial_data):
    """Wipe and rebuild the foreground database for the given scenario.

    Parameters
    ----------
    scenario : Scenario
        Exchange amounts for this scenario (from src.scenarios).
    spatial_data : dict
        Keys: 'tz_districts_gdf', 'production_weights', 'ginning_weights',
        'textile_weights'.

    Returns
    -------
    None  (all data is persisted to the Brightway database on disk)
    """
    tz_districts_gdf  = spatial_data["tz_districts_gdf"]
    production_weights = spatial_data["production_weights"]
    ginning_weights    = spatial_data["ginning_weights"]
    textile_weights    = spatial_data["textile_weights"]

    # 1. Clear and recreate the foreground database ───────────────────────────
    if DB_FOREGROUND in bd.databases:
        del bd.databases[DB_FOREGROUND]
        print(f"Cleared existing '{DB_FOREGROUND}' database.")
    foreground = bd.Database(DB_FOREGROUND)
    foreground.register()

    # 2. Resolve biosphere flows ──────────────────────────────────────────────
    biosphere = bd.Database(DB_BIOSPHERE)

    landuse_agricultural = _get_bio_node(
        biosphere, "Occupation, permanent crop, non-irrigated, intensive",
        ("natural resource", "land"))
    landuse_industrial = _get_bio_node(
        biosphere, "Occupation, industrial area",
        ("natural resource", "land"))
    water_to_air     = _get_bio_node(biosphere, "Water",       ("air",))
    water_from_river = _get_bio_node(biosphere, "Water, river",
                                     ("natural resource", "in water"))
    nitrite_emission     = _get_bio_node(biosphere, "Nitrite",
                                         ("water", "surface water"))
    phosphorous_emission = _get_bio_node(biosphere, "Phosphorus",
                                         ("water", "surface water"))
    pesticide_emission   = _get_bio_node(biosphere, "Pesticides, unspecified",
                                         ("soil", "agricultural"))

    # Field GHG emissions from fertilizer application:
    #   • N₂O (direct + indirect) from N applied (synthetic and organic)
    #   • CO₂ from urea hydrolysis (urea → NH₄⁺ + CO₂ in soil)
    # Both are characterized by the existing climate-change LCIA method.
    # Field N₂O and urea-CO₂ are best modeled as unspecified-air emissions;
    # the LC-IMPACT NaturalEarth method characterizes "Emissions to air,
    # unspecified" → maps to biosphere3 ("air",). That's the cleanest match
    # to the IPCC-Tier-1 EFs which don't distinguish high-stack vs ground-level.
    n2o_air = _get_bio_node_first(biosphere, "Dinitrogen monoxide",
        [("air",),
         ("air", "non-urban air or from high stacks"),
         ("air", "low population density, long-term")])
    co2_fossil_air = _get_bio_node_first(biosphere, "Carbon dioxide, fossil",
        [("air",),
         ("air", "non-urban air or from high stacks"),
         ("air", "low population density, long-term")])

    # Pesticide-specific biosphere flows. The OLCA-Pest fractions split the
    # applied AI across four environmental compartments (air, agricultural
    # soil, natural/forest soil, surface water). For each pesticide we pull
    # all four compartment-specific flows from biosphere3, falling back to
    # less-specific compartments where biosphere3 doesn't carry the precise
    # sub-compartment.
    profenofos_air = _get_bio_node_first(biosphere, "Profenofos",
        [("air", "low population density"), ("air", "non-urban air or from high stacks"), ("air",)])
    profenofos_soil_agri = _get_bio_node(biosphere, "Profenofos",
                                         ("soil", "agricultural"))
    profenofos_soil_nat = _get_bio_node_first(biosphere, "Profenofos",
        [("soil", "forestry"), ("soil",)])
    profenofos_water = _get_bio_node_first(biosphere, "Profenofos",
        [("water", "surface water"), ("water",)])

    lambda_air = _get_bio_node_first(biosphere, "Lambda-cyhalothrin",
        [("air", "low population density"), ("air", "non-urban air or from high stacks"), ("air",)])
    lambda_soil_agri = _get_bio_node(biosphere, "Lambda-cyhalothrin",
                                     ("soil", "agricultural"))
    lambda_soil_nat = _get_bio_node_first(biosphere, "Lambda-cyhalothrin",
        [("soil", "forestry"), ("soil",)])
    lambda_water = _get_bio_node_first(biosphere, "Lambda-cyhalothrin",
        [("water", "surface water"), ("water",)])

    # 3. Resolve ecoinvent background processes ───────────────────────────────
    ecoinvent = bd.Database(DB_ECOINVENT)

    def _ei(name, location, ref_product, unit):
        matches = [a for a in ecoinvent
                   if a["name"] == name
                   and a["location"] == location
                   and a["reference product"] == ref_product
                   and a["unit"] == unit]
        if not matches:
            raise ValueError(f"ecoinvent node not found: {name!r} @ {location}")
        return matches[0]

    pesticide_input   = _ei("market for pesticide, unspecified",
                            "GLO", "pesticide, unspecified", "kilogram")
    tractor_input     = _ei("market for tractor, 4-wheel, agricultural",
                            "GLO", "tractor, 4-wheel, agricultural", "kilogram")
    fertilizer_input_N = _ei("nutrient supply from manure, solid, cattle",
                             "GLO", "organic nitrogen fertiliser, as N", "kilogram")
    # Synthetic N — urea is the dominant synthetic fertilizer used by
    # conventional Tanzanian cotton smallholders. Dosed in kg-urea-as-product;
    # convert from kg N via _UREA_N_CONTENT.
    urea_input        = _ei("market for urea",
                            "RER", "urea", "kilogram")
    fertilizer_input_P = _ei("nutrient supply from manure, solid, cattle",
                             "GLO", "organic phosphorus fertiliser, as P2O5", "kilogram")
    electricity_input  = _ei("market for electricity, medium voltage",
                             "TZ", "electricity, medium voltage", "kilowatt hour")
    transportation_lorry = _ei("market for transport, freight, lorry, unspecified",
                               "RoW",
                               "transport, freight, lorry, diesel, unspecified",
                               "ton kilometer")
    waste_cotton_management = _ei(
        "treatment of waste yarn and waste textile, unsanitary landfill",
        "RoW", "waste yarn and waste textile", "kilogram")
    building_construction = _ei(
        "building construction, hall, steel construction",
        "RoW", "building, hall, steel construction", "square meter")
    coal_input = _ei("electricity production, hard coal, conventional",
                     "ZA", "electricity, high voltage", "kilowatt hour")
    maize_starch_input = _ei("market for maize starch",
                             "GLO", "maize starch", "kilogram")
    wastewater_management_textile = _ei(
        "market for wastewater from textile production",
        "RoW", "wastewater from textile production", "cubic meter")
    oil_mill_construction = _ei("market for oil mill",
                                "GLO", "oil mill", "unit")
    waste_management_oil_mill = _ei("market for municipal solid waste",
                                    "RoW", "municipal solid waste", "kilogram")
    wastewater_management_oil_mill = _ei(
        "market for wastewater from vegetable oil refinery",
        "GLO", "wastewater from vegetable oil refinery", "cubic meter")

    # 4. Create product nodes ─────────────────────────────────────────────────
    def _prod(code, name, unit="kilogram", location="TZ"):
        n = foreground.new_node(code=code, name=name, unit=unit,
                                location=location,
                                type=bd.labels.product_node_default)
        n.save()
        return n

    raw_cotton           = _prod("prod1",   "raw_cotton")
    cotton_lint          = _prod("prod2_1", "cotton_lint")
    cotton_seed          = _prod("prod2_2", "cotton_seed")
    textile              = _prod("prod3",   "textile")
    seed_cake            = _prod("prod6_1", "seed_cake")
    seed_oil             = _prod("prod6_2", "seed_oil")
    foreign_consumption  = _prod("prod7",   "foreign_consumption")
    domestic_consumption = _prod("prod8",   "domestic_consumption")

    # 5. Create national process stub nodes ───────────────────────────────────
    def _proc(code, name, location="TZ"):
        n = foreground.new_node(code=code, name=name, location=location,
                                type=bd.labels.process_node_default)
        n.save()
        return n

    cotton_production  = _proc("act1",  "cotton_production")
    lint_ginning       = _proc("act2a", "lint_ginning")
    seed_ginning       = _proc("act2b", "seed_ginning")
    textile_production = _proc("act3",  "textile_production")
    oil_milling_oil    = _proc("act6a", "oil_milling_oil")
    oil_milling_cake   = _proc("act6b", "oil_milling_cake")
    export_market      = _proc("act7",  "export_market")
    domestic_market    = _proc("act8",  "domestic_market")

    # 6. Non-regionalized edges (oil milling, markets) ────────────────────────
    sc = scenario  # shorthand

    # Oil milling — oil fraction (OIL_ALLOC ~24%)
    oil_milling_oil.new_edge(amount=sc.oil_seed_input_kg * OIL_ALLOC,   type=bd.labels.consumption_edge_default, input=cotton_seed).save()
    oil_milling_oil.new_edge(amount=sc.oil_water_air_m3 * OIL_ALLOC,    type=bd.labels.biosphere_edge_default,   input=water_to_air).save()
    oil_milling_oil.new_edge(amount=sc.oil_water_river_m3 * OIL_ALLOC,  type=bd.labels.biosphere_edge_default,   input=water_from_river).save()
    oil_milling_oil.new_edge(amount=sc.oil_electricity_kwh * OIL_ALLOC, type=bd.labels.consumption_edge_default, input=electricity_input, **_ln_unc(sc.oil_electricity_kwh * OIL_ALLOC, "oil_electricity_kwh")).save()
    oil_milling_oil.new_edge(amount=sc.oil_mill_unit * OIL_ALLOC,       type=bd.labels.consumption_edge_default, input=oil_mill_construction).save()
    oil_milling_oil.new_edge(amount=sc.oil_waste_kg * OIL_ALLOC,        type=bd.labels.consumption_edge_default, input=waste_management_oil_mill).save()
    oil_milling_oil.new_edge(amount=sc.oil_wastewater_m3 * OIL_ALLOC,   type=bd.labels.consumption_edge_default, input=wastewater_management_oil_mill).save()
    oil_milling_oil.new_edge(amount=0.308411215,                         type=bd.labels.production_edge_default,  input=seed_oil).save()

    # Oil milling — cake fraction (CAKE_ALLOC ~76%)
    oil_milling_cake.new_edge(amount=sc.oil_seed_input_kg * CAKE_ALLOC,   type=bd.labels.consumption_edge_default, input=cotton_seed).save()
    oil_milling_cake.new_edge(amount=sc.oil_water_air_m3 * CAKE_ALLOC,    type=bd.labels.biosphere_edge_default,   input=water_to_air).save()
    oil_milling_cake.new_edge(amount=sc.oil_water_river_m3 * CAKE_ALLOC,  type=bd.labels.biosphere_edge_default,   input=water_from_river).save()
    oil_milling_cake.new_edge(amount=sc.oil_electricity_kwh * CAKE_ALLOC, type=bd.labels.consumption_edge_default, input=electricity_input, **_ln_unc(sc.oil_electricity_kwh * CAKE_ALLOC, "oil_electricity_kwh")).save()
    oil_milling_cake.new_edge(amount=sc.oil_mill_unit * CAKE_ALLOC,       type=bd.labels.consumption_edge_default, input=oil_mill_construction).save()
    oil_milling_cake.new_edge(amount=sc.oil_waste_kg * CAKE_ALLOC,        type=bd.labels.consumption_edge_default, input=waste_management_oil_mill).save()
    oil_milling_cake.new_edge(amount=sc.oil_wastewater_m3 * CAKE_ALLOC,   type=bd.labels.consumption_edge_default, input=wastewater_management_oil_mill).save()
    oil_milling_cake.new_edge(amount=1,                                    type=bd.labels.production_edge_default,  input=seed_cake).save()

    # Export market
    export_market.new_edge(amount=1,                          type=bd.labels.production_edge_default,  input=foreign_consumption).save()
    export_market.new_edge(amount=1,                          type=bd.labels.consumption_edge_default, input=cotton_lint).save()
    export_market.new_edge(amount=0,                          type=bd.labels.consumption_edge_default, input=seed_cake).save()
    export_market.new_edge(amount=0,                          type=bd.labels.consumption_edge_default, input=seed_oil).save()
    export_market.new_edge(amount=0,                          type=bd.labels.consumption_edge_default, input=textile).save()
    export_market.new_edge(amount=sc.export_transport_tkm,   type=bd.labels.consumption_edge_default, input=transportation_lorry).save()

    # Domestic market (weighted basket: seed_cake, seed_oil, textile)
    # Basket fractions are scenario-specific so the manufacturing-expansion
    # scenarios get a textile-heavier basket (and high-yield scenarios
    # don't, since textile capacity is saturated). Combined with a
    # scenario-scaled total demand in the notebook, this propagates
    # yield + manufacturing levers to absolute LCA impacts.
    basket_cake, basket_oil, basket_textile = sc.domestic_basket_fractions()
    domestic_market.new_edge(amount=1,              type=bd.labels.production_edge_default,  input=domestic_consumption).save()
    domestic_market.new_edge(amount=basket_cake,    type=bd.labels.consumption_edge_default, input=seed_cake).save()
    domestic_market.new_edge(amount=basket_oil,     type=bd.labels.consumption_edge_default, input=seed_oil).save()
    domestic_market.new_edge(amount=basket_textile, type=bd.labels.consumption_edge_default, input=textile).save()
    domestic_market.new_edge(amount=0,              type=bd.labels.consumption_edge_default, input=transportation_lorry).save()

    print("Non-regionalized edges created.")

    # 7. District cotton production activities ────────────────────────────────
    district_cotton_activities = {}

    for district_id, weight in production_weights.items():
        row = tz_districts_gdf[tz_districts_gdf["ADM2_PCODE"] == district_id]
        district_name = row["ADM2_EN"].values[0] if len(row) else district_id

        act = foreground.new_node(
            code=f"cotton_prod_{district_id}",
            name=f"cotton production, {district_name}",
            location=("tz_districts", district_id),
            type=bd.labels.process_node_default,
        )

        act.new_edge(amount=1,               type=bd.labels.production_edge_default,  input=act).save()
        act.new_edge(amount=sc.seed_replant_kg, type=bd.labels.consumption_edge_default, input=cotton_seed).save()

        act.new_edge(
            amount=sc.land_occupation_m2,
            type=bd.labels.biosphere_edge_default, input=landuse_agricultural,
            **_ln_unc(sc.land_occupation_m2, "land_occupation_m2"),
        ).save()
        # Pesticide biosphere emissions: split each AI across the 4 OLCA-Pest
        # environmental compartments (air, agricultural soil, natural soil,
        # surface water). Lognormal uncertainty applied to the agricultural-
        # soil emission (the dominant fraction). The crop-uptake fraction
        # (~0.799) is not an environmental emission and is omitted here.
        for ai_kg, flow_air, flow_soil_a, flow_soil_n, flow_water in [
            (sc.profenofos_kg,
             profenofos_air, profenofos_soil_agri, profenofos_soil_nat, profenofos_water),
            (sc.lambda_cyhalothrin_kg,
             lambda_air, lambda_soil_agri, lambda_soil_nat, lambda_water),
        ]:
            act.new_edge(amount=ai_kg * _OLCA_F_AIR,
                         type=bd.labels.biosphere_edge_default, input=flow_air).save()
            act.new_edge(
                amount=ai_kg * _OLCA_F_SOIL_AGRI,
                type=bd.labels.biosphere_edge_default, input=flow_soil_a,
                **_ln_unc(ai_kg * _OLCA_F_SOIL_AGRI, "pesticide_kg"),
            ).save()
            act.new_edge(amount=ai_kg * _OLCA_F_SOIL_NAT,
                         type=bd.labels.biosphere_edge_default, input=flow_soil_n).save()
            act.new_edge(amount=ai_kg * _OLCA_F_WATER_SURF,
                         type=bd.labels.biosphere_edge_default, input=flow_water).save()
        act.new_edge(amount=sc.n_emission_kg, type=bd.labels.biosphere_edge_default,
                     input=nitrite_emission, **_ln_unc(sc.n_emission_kg, "n_emission_kg")).save()
        act.new_edge(amount=sc.p_emission_kg, type=bd.labels.biosphere_edge_default,
                     input=phosphorous_emission, **_ln_unc(sc.p_emission_kg, "p_emission_kg")).save()

        # Field GHG emissions from N fertilizer (IPCC 2019, dry climate):
        #   • Direct N₂O applies EF 0.6 % to manure-N and 1.0 % to urea-N separately
        #   • Indirect N₂O (volatilisation + leaching) applies uniformly to total N
        #   • CO₂ from urea hydrolysis applies to urea-N only (synthetic share only)
        n2o_kg = (
            sc.fertilizer_n_kg_manure * (_EF_N2O_DIRECT_ORG  + _EF_N2O_INDIRECT)
            + sc.fertilizer_n_kg_urea * (_EF_N2O_DIRECT_SYNTH + _EF_N2O_INDIRECT)
        ) * _N2O_PER_N
        urea_kg = sc.fertilizer_n_kg_urea / _UREA_N_CONTENT
        co2_urea_kg = urea_kg * _UREA_CO2_EF
        # N2O is a weighted SUM of manure-N and urea-N terms, so lognormal
        # scale-invariance does not cleanly transfer a single parameter's sigma;
        # left deterministic (the fertilizer technosphere inputs below carry the
        # N uncertainty). CO2 derives purely from urea, so it inherits the
        # fertilizer_n_kg_urea sigma.
        act.new_edge(amount=n2o_kg,
                     type=bd.labels.biosphere_edge_default, input=n2o_air).save()
        act.new_edge(amount=co2_urea_kg, type=bd.labels.biosphere_edge_default,
                     input=co2_fossil_air, **_ln_unc(co2_urea_kg, "fertilizer_n_kg_urea")).save()

        act.new_edge(
            amount=sc.pesticide_kg,
            type=bd.labels.consumption_edge_default, input=pesticide_input,
            **_ln_unc(sc.pesticide_kg, "pesticide_kg"),
        ).save()
        # Tractor: scenario field gives tractors-per-kg-raw-cotton (count fraction).
        # The ecoinvent reference product is dosed per kg-of-tractor, so multiply
        # by mean tractor mass (default 2000 kg) to get the technosphere amount.
        act.new_edge(amount=sc.tractor_kg * sc.tractor_mass_kg,
                     type=bd.labels.consumption_edge_default, input=tractor_input,
                     **_ln_unc(sc.tractor_kg * sc.tractor_mass_kg, "tractor_kg")).save()
        # Irrigation water withdrawal (zero in TZ baseline; non-zero scenarios
        # exercise this edge via sc.irrigation_water_m3). Reuses the
        # water_from_river biosphere flow already resolved for textile/oil-mill.
        if sc.irrigation_water_m3 > 0:
            act.new_edge(amount=sc.irrigation_water_m3,
                         type=bd.labels.biosphere_edge_default,
                         input=water_from_river).save()
        # N fertilizer: split blended N into organic-manure and synthetic-urea
        # technosphere paths so upstream burdens (Haber-Bosch for urea, manure
        # supply chain for organic) are captured in the LCA matrix.
        act.new_edge(amount=sc.fertilizer_n_kg_manure,
                     type=bd.labels.consumption_edge_default, input=fertilizer_input_N,
                     **_ln_unc(sc.fertilizer_n_kg_manure, "fertilizer_n_kg_manure")).save()
        act.new_edge(amount=urea_kg,
                     type=bd.labels.consumption_edge_default, input=urea_input,
                     **_ln_unc(urea_kg, "fertilizer_n_kg_urea")).save()
        act.new_edge(amount=sc.fertilizer_p_kg, type=bd.labels.consumption_edge_default,
                     input=fertilizer_input_P, **_ln_unc(sc.fertilizer_p_kg, "fertilizer_p_kg")).save()

        act.save()
        district_cotton_activities[district_id] = act

    print(f"Created {len(district_cotton_activities)} district cotton production activities.")

    # 8. Cotton production national aggregator ────────────────────────────────
    cotton_production = [a for a in foreground if a["code"] == "act1"][0]
    for exc in list(cotton_production.exchanges()):
        exc.delete()

    cotton_production.new_edge(amount=1, type=bd.labels.production_edge_default, input=raw_cotton).save()
    for district_id, weight in production_weights.items():
        cotton_production.new_edge(
            amount=weight,
            type=bd.labels.consumption_edge_default,
            input=district_cotton_activities[district_id],
        ).save()
    print(f"cotton_production aggregates {len(production_weights)} districts.")

    # 9. District ginning activities ──────────────────────────────────────────
    district_lint_ginning = {}
    district_seed_ginning = {}

    for district_id, weight in ginning_weights.items():
        row = tz_districts_gdf[tz_districts_gdf["ADM2_PCODE"] == district_id]
        district_name = row["ADM2_EN"].values[0] if len(row) else district_id

        # Lint ginning (LINT_ALLOC ~37%)
        lint_act = foreground.new_node(
            code=f"lint_gin_{district_id}",
            name=f"lint ginning, {district_name}",
            location=("tz_districts", district_id),
            type=bd.labels.process_node_default,
        )
        lint_act.new_edge(amount=1,                                              type=bd.labels.production_edge_default,  input=lint_act).save()
        lint_act.new_edge(amount=2.74122807 * LINT_ALLOC,                        type=bd.labels.consumption_edge_default, input=raw_cotton).save()
        lint_act.new_edge(amount=sc.ginning_land_m2 * LINT_ALLOC,               type=bd.labels.biosphere_edge_default,   input=landuse_industrial).save()
        lint_act.new_edge(amount=sc.ginning_electricity_kwh * LINT_ALLOC,       type=bd.labels.consumption_edge_default, input=electricity_input,     **_ln_unc(sc.ginning_electricity_kwh * LINT_ALLOC, "ginning_electricity_kwh")).save()
        lint_act.new_edge(amount=sc.ginning_transport_tkm * LINT_ALLOC,         type=bd.labels.consumption_edge_default, input=transportation_lorry,   **_ln_unc(sc.ginning_transport_tkm * LINT_ALLOC, "ginning_transport_tkm")).save()
        lint_act.new_edge(amount=sc.ginning_waste_kg * LINT_ALLOC,              type=bd.labels.consumption_edge_default, input=waste_cotton_management).save()
        lint_act.new_edge(amount=sc.ginning_building_m2 * LINT_ALLOC,           type=bd.labels.consumption_edge_default, input=building_construction).save()
        lint_act.save()
        district_lint_ginning[district_id] = lint_act

        # Seed ginning (SEED_ALLOC ~63%)
        seed_act = foreground.new_node(
            code=f"seed_gin_{district_id}",
            name=f"seed ginning, {district_name}",
            location=("tz_districts", district_id),
            type=bd.labels.process_node_default,
        )
        seed_act.new_edge(amount=1,                                              type=bd.labels.production_edge_default,  input=seed_act).save()
        seed_act.new_edge(amount=2.74122807 * SEED_ALLOC,                        type=bd.labels.consumption_edge_default, input=raw_cotton).save()
        seed_act.new_edge(amount=sc.ginning_land_m2 * SEED_ALLOC,               type=bd.labels.biosphere_edge_default,   input=landuse_industrial).save()
        seed_act.new_edge(amount=sc.ginning_electricity_kwh * SEED_ALLOC,       type=bd.labels.consumption_edge_default, input=electricity_input,     **_ln_unc(sc.ginning_electricity_kwh * SEED_ALLOC, "ginning_electricity_kwh")).save()
        seed_act.new_edge(amount=sc.ginning_transport_tkm * SEED_ALLOC,         type=bd.labels.consumption_edge_default, input=transportation_lorry,   **_ln_unc(sc.ginning_transport_tkm * SEED_ALLOC, "ginning_transport_tkm")).save()
        seed_act.new_edge(amount=sc.ginning_waste_kg * SEED_ALLOC,              type=bd.labels.consumption_edge_default, input=waste_cotton_management).save()
        seed_act.new_edge(amount=sc.ginning_building_m2 * SEED_ALLOC,           type=bd.labels.consumption_edge_default, input=building_construction).save()
        seed_act.save()
        district_seed_ginning[district_id] = seed_act

    print(f"Created {len(district_lint_ginning)} district lint/seed ginning activities.")

    # 10. Ginning national aggregators ────────────────────────────────────────
    lint_ginning = [a for a in foreground if a["code"] == "act2a"][0]
    for exc in list(lint_ginning.exchanges()):
        exc.delete()
    lint_ginning.new_edge(amount=1, type=bd.labels.production_edge_default, input=cotton_lint).save()
    for district_id, weight in ginning_weights.items():
        lint_ginning.new_edge(amount=weight, type=bd.labels.consumption_edge_default,
                              input=district_lint_ginning[district_id]).save()

    seed_ginning = [a for a in foreground if a["code"] == "act2b"][0]
    for exc in list(seed_ginning.exchanges()):
        exc.delete()
    seed_ginning.new_edge(amount=1.699561404, type=bd.labels.production_edge_default, input=cotton_seed).save()
    for district_id, weight in ginning_weights.items():
        seed_ginning.new_edge(amount=weight, type=bd.labels.consumption_edge_default,
                              input=district_seed_ginning[district_id]).save()

    print(f"Ginning aggregators built ({len(ginning_weights)} districts).")

    # 11. District textile production activities ───────────────────────────────
    district_textile = {}

    for district_id, weight in textile_weights.items():
        row = tz_districts_gdf[tz_districts_gdf["ADM2_PCODE"] == district_id]
        district_name = row["ADM2_EN"].values[0] if len(row) else district_id

        act = foreground.new_node(
            code=f"textile_{district_id}",
            name=f"textile production, {district_name}",
            location=("tz_districts", district_id),
            type=bd.labels.process_node_default,
        )

        act.new_edge(amount=1,                           type=bd.labels.production_edge_default,  input=act).save()
        # textile_lint_kg is a fixed mass-conversion coefficient (no uncertainty).
        act.new_edge(amount=sc.textile_lint_kg,
                     type=bd.labels.consumption_edge_default, input=cotton_lint).save()
        act.new_edge(amount=sc.textile_land_m2, type=bd.labels.biosphere_edge_default, input=landuse_industrial).save()
        act.new_edge(
            amount=sc.textile_water_air_m3,
            type=bd.labels.biosphere_edge_default, input=water_to_air,
            **_ln_unc(sc.textile_water_air_m3, "textile_water_air_m3"),
        ).save()
        act.new_edge(
            amount=sc.textile_water_river_m3,
            type=bd.labels.biosphere_edge_default, input=water_from_river,
            **_ln_unc(sc.textile_water_river_m3, "textile_water_river_m3"),
        ).save()
        act.new_edge(amount=sc.textile_electricity_kwh, type=bd.labels.consumption_edge_default,
                     input=electricity_input, **_ln_unc(sc.textile_electricity_kwh, "textile_electricity_kwh")).save()
        act.new_edge(
            amount=sc.textile_coal_kwh,
            type=bd.labels.consumption_edge_default, input=coal_input,
            **_ln_unc(sc.textile_coal_kwh, "textile_coal_kwh"),
        ).save()
        act.new_edge(amount=sc.textile_building_m2, type=bd.labels.consumption_edge_default, input=building_construction).save()
        act.new_edge(amount=sc.textile_starch_kg,   type=bd.labels.consumption_edge_default, input=maize_starch_input).save()
        # textile_waste_kg is well documented (no uncertainty).
        act.new_edge(amount=sc.textile_waste_kg,
                     type=bd.labels.consumption_edge_default, input=waste_cotton_management).save()
        act.new_edge(amount=sc.textile_wastewater_m3, type=bd.labels.consumption_edge_default, input=wastewater_management_textile).save()

        act.save()
        district_textile[district_id] = act

    print(f"Created {len(district_textile)} district textile production activities.")

    # 12. Textile production national aggregator ───────────────────────────────
    textile_production = [a for a in foreground if a["code"] == "act3"][0]
    for exc in list(textile_production.exchanges()):
        exc.delete()
    textile_production.new_edge(amount=1, type=bd.labels.production_edge_default, input=textile).save()
    for district_id, weight in textile_weights.items():
        textile_production.new_edge(amount=weight, type=bd.labels.consumption_edge_default,
                                    input=district_textile[district_id]).save()

    print(f"textile_production aggregates {len(textile_weights)} districts.")

    # 13. Set foreground geocollections metadata ───────────────────────────────
    foreground = bd.Database(DB_FOREGROUND)
    foreground.metadata["geocollections"] = ["tz_districts"]

    # 14. Label foreground geocollections ─────────────────────────────────────
    import bw2regional as bwr
    bwr.label_activity_geocollections(DB_FOREGROUND)
    print(f"Foreground geocollections: {bd.databases[DB_FOREGROUND]['geocollections']}")

    print(f"\n[build_foreground_db] Scenario '{scenario.name}' complete — "
          f"{len(list(foreground))} nodes in foreground DB.")
