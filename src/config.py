"""
config.py — Project-wide constants and file paths for the Tanzania Cotton LCA.

All paths, database names, method names, and fixed physical constants live here.
Exchange amounts that vary between scenarios live in scenarios.py.
"""

import os

# ── Brightway project / database names ──────────────────────────────────────
PROJECT_NAME = "tz_cotton"
DB_BIOSPHERE = "biosphere-3.12"
DB_ECOINVENT = "ecoinvent-3.12-cutoff"
DB_FOREGROUND = "foreground"

BACKUP_PATH = "X:/Eli/projects/tz_cotton/data_ecoinvent/backup/ecoinvent.tar.gz"

# ── Spatial dataset paths ────────────────────────────────────────────────────
COTTON_RASTER_PATH = "X:/Eli/projects/tz_cotton/data_spatial/cotton_production_tz/spam_2020_P_cotton_A_tz.tif"
# CF_domain.csv aggregates the five LC-IMPACT species groups (amphibians,
# birds, mammals, plants, reptiles) into a single "Eukaryota" domain, giving
# exactly one row per (ecoregion, habitat). CF.csv is resolved per species
# group instead, and habitat coverage differs between groups, which made any
# average across it mix incommensurable taxa.
LAND_USE_CFS_PATH = "X:/Eli/data/cf/land use/CF_domain.csv"
WATER_CFS_PATH = "X:/Eli/data/cf/water consumption/water.xlsx"
WATER_BASINS_PATH = "X:/Eli/data/cf/water consumption/1-s2.0-S0048969722058016-mmc3/basins_5min_pcrglobwb.gpkg"
N_CFS_PATH = "X:/Eli/data/cf/freshwater eutrophication/CFs_freshwater_eutrophication/global_species_loss/N/ASCII_rasters/CF_marginal_diffuse.asc"
P_CFS_PATH = "X:/Eli/data/cf/freshwater eutrophication/CFs_freshwater_eutrophication/global_species_loss/P/ASCII_rasters/CF_marginal_diffuse.asc"
# Climate-change CF data was reorganised into metric subfolders (pdf_richness /
# msa / functional_diversity); this is the species-richness (PDF) endpoint file.
CC_CFS_PATH = "X:/Eli/data/cf/climate change/pdf_richness/EQ_ClimateChange_Terrestrial_gloPDF_Country_Aggregated_2025-12-09.xlsx"
ECOTOX_CFS_PATH = "X:/Eli/data/cf/ecotoxicity/USEtox2.1_LC-Impact_v2_results/USEtox2.1_LC-Impact_v2_results_ECOTOX_20190326.xlsx"
TZ_DISTRICTS_PATH = "X:/Eli/data/shapefiles/Administrative Regions TZ/districts/tza_admbnda_adm2_20181019.shp"
WWF_ECOREGIONS_PATH = "X:/Eli/data/shapefiles/TEOW/wwf_terr_ecos.shp"
GINNERIES_PATH = (
    "X:/Eli/projects/tz_cotton/data_spatial/ginneries/ginnery_locations_and_outputs.shp"
)
TEXTILE_PLANTS_PATH = (
    "X:/Eli/projects/tz_cotton/data_spatial/textiles/textiles_locations.shp"
)

# ── Output paths ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # python/ folder
RESULTS_TABLES_DIR = os.path.join(_HERE, "results", "tables")
RESULTS_FIGURES_DIR = os.path.join(_HERE, "results", "figures")

# ── Mass allocation factors ──────────────────────────────────────────────────
# Ginning: 1 kg lint + 1.700 kg seed from 2.700 kg raw cotton (mass allocation)
# Oil milling: 0.308 kg seed oil + 1.000 kg seed cake from 1.308 kg cottonseed
LINT_ALLOC = 1 / 2.699561404  # 0.370449 (37% of ginning burden to lint)
SEED_ALLOC = 1.699561404 / 2.699561404  # 0.629551 (63% of ginning burden to seed)
OIL_ALLOC = 0.308411215 / 1.308411215  # 0.235697 (24% of oil-milling burden to oil)
CAKE_ALLOC = 1 / 1.308411215  # 0.764303 (76% of oil-milling burden to cake)

# ── Annual industry volumes for the combined demand vector ───────────────────
# Source: LCA_System sheet, cotton_industry_data.xlsx (tonnes/year)
DOMESTIC_SEED_CAKE_T = 80333.440954
DOMESTIC_SEED_OIL_T = 24775.734126
DOMESTIC_TEXTILE_T = 17159.408311
EXPORT_LINT_T = 82447.607501
DOMESTIC_TOTAL_T = DOMESTIC_SEED_CAKE_T + DOMESTIC_SEED_OIL_T + DOMESTIC_TEXTILE_T

# Domestic market basket fractions (used as edge amounts in domestic_market activity)
BASKET_SEED_CAKE = DOMESTIC_SEED_CAKE_T / DOMESTIC_TOTAL_T
BASKET_SEED_OIL = DOMESTIC_SEED_OIL_T / DOMESTIC_TOTAL_T
BASKET_TEXTILE = DOMESTIC_TEXTILE_T / DOMESTIC_TOTAL_T

# ── LCIA method name tuples ──────────────────────────────────────────────────
METHOD_LAND_USE_REGIONAL = (
    "LC-IMPACT",
    "Land Use",
    "Occupation",
    "Regionalized Districts TZ",
)
METHOD_LAND_USE_GENERIC = (
    "LC-IMPACT",
    "Land Use",
    "Occupation",
    "Average",
    "Site-generic",
)
METHOD_WATER = (
    "LC-IMPACT",
    "Water Consumption",
    "Freshwater Ecosystem Quality",
    "Regionalized Districts TZ",
)
METHOD_N_EUTRO = (
    "Freshwater Eutrophication",
    "Nitrogen",
    "Species Loss",
    "Regionalized Districts TZ",
)
METHOD_P_EUTRO = (
    "Freshwater Eutrophication",
    "Phosphorus",
    "Species Loss",
    "Regionalized Districts TZ",
)
METHOD_CLIMATE_CHANGE = (
    "LC-IMPACT",
    "Climate Change",
    "Terrestrial Biodiversity",
    "Global rcp26",
)
METHOD_ECOTOX_FW = (
    "USEtox 2.1 / LC-IMPACT",
    "Ecotoxicity",
    "Freshwater",
    "W6 N-W-E-C Africa, metals 100yr, no-LT",
)
# Earlier registrations of the same method, deregistered by 00_setup on rebuild.
METHOD_ECOTOX_FW_SUPERSEDED = [
    ("USEtox 2.1 / LC-IMPACT", "Ecotoxicity", "Freshwater", "Long-term, global average"),
    ("USEtox 2.1 / LC-IMPACT", "Ecotoxicity", "Freshwater", "Long-term, W6 N-W-E-C Africa"),
]

# ── Ecotoxicity: time horizon and long-term emissions ─────────────────────
# Two unrelated senses of "long-term" meet in this method. Keep them apart.
#
# 1. IMPACT horizon — which workbook sheet a CF is read from.
#    "all impacts, long-term" integrates to infinity and is the ONLY sheet
#    carrying organics: profenofos, lambda-cyhalothrin and every other
#    pesticide appear nowhere else. "all impacts, 100 years" holds just the
#    27 metals, which are the substances for which the horizon actually bites
#    — infinite-horizon metal CFs run 3.5-130x the 100-year values (Al(III):
#    62.6x). We therefore read metals from the 100-year sheet and everything
#    else from the long-term sheet.
#    NB: the workbook's own "About" text describes the two sheets the wrong
#    way round. Trust the contents, not the description.
#
# 2. EMISSION timing — the ecoinvent subcompartment suffix ", long-term",
#    meaning release occurs >100 yr after the activity (landfill and slag-heap
#    leachate). ecoinvent takes no position on these and ships both variants;
#    this study excludes them ("no-LT"). Characterized, they were 98.8% of the
#    baseline ecotoxicity score — 98.7% from Al(III) to
#    ('water', 'ground-, long-term') alone — which buried the foreground
#    pesticide signal about six orders of magnitude below a background
#    coal-ash leachate integral. The excluded flows are ~100% metals;
#    organics in long-term compartments are ~8e-16 of the score.
ECOTOX_SHEET_DEFAULT = "all impacts, long-term"   # organics + metals, t = inf
ECOTOX_SHEET_METALS  = "all impacts, 100 years"   # 27 metals only, t = 100 yr
ECOTOX_LT_MARKER     = "long-term"                # subcompartment suffix to drop

# ── USEtox 2.1 / LC-IMPACT spatial parameters for Tanzania (W6 zone) ─────────
# Tanzania is located in East Africa → W6 (North, West, East & Central Africa)
# is the appropriate continental box in the USEtox 2.1 / LC-IMPACT v2 model.
#
# V_FW_W6_M3: freshwater compartment volume for the W6 continental box.
#   This value is not listed directly in the USEtox results spreadsheet.
#   It is embedded in the fate factors via the hydrological residence time.
#   Estimated at ~1 × 10¹⁰ m³ (order-of-magnitude from iFF-derived residence
#   time × African continental river discharge; excludes deep stratified lakes).
#   Confirm against LC-IMPACT v2 technical report / supplementary information
#   if precise normalization is required for absolute reporting.
#
# ECOTOX_M3DAY_TO_PDFYR: converts the LCA score unit (PDF·m³·day) to
#   (PDF·yr), i.e. the fraction of species affected uniformly across the W6
#   freshwater volume over one year. Used only for display/reporting;
#   relative scenario rankings are unaffected.
USETOX_CONTINENT_LABEL = "W6 (North, West, East & Central Africa)"
USETOX_V_FW_W6_M3 = 1.0e10   # m³ (estimate — see note above)
ECOTOX_M3DAY_TO_PDFYR = 1.0 / (USETOX_V_FW_W6_M3 * 365.25)  # ≈ 2.74e-13

# ── Geocollection and extension table names ──────────────────────────────────
GC_TZ_DISTRICTS = "tz_districts"
GC_WWF_ECOREGION = "wwf_ecoregions"
GC_COTTON_RASTER = "cotton_production_raster"
XT_COTTON_PRODUCTION = "cotton_production_output"
