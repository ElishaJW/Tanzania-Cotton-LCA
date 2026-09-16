"""
monte_carlo.py
Monte Carlo uncertainty propagation for the Tanzania cotton LCA.

WHAT IT DOES
------------
For each scenario and each of the six endpoint impact categories, it resamples
every exchange that carries a stats_arrays distribution (the foreground
lognormals defined in src/inventory.py, PLUS ecoinvent's own background
pedigree distributions) N times, re-solving the LCA each iteration. The result
is an empirical distribution of the score per (scenario, category).

The deterministic pipeline (01_scenarios.ipynb) is untouched: this writes
SEPARATE tables and the plotting scripts keep using the deterministic means.

HOW IT INTERACTS WITH THE OTHER SCRIPTS
---------------------------------------
  src/scenarios.py    -> Scenario objects (central values)
  src/inventory.py    -> build_foreground_db() + the _SIGMA_LN foreground
                         distributions; this script adds NO new uncertainty,
                         it only propagates what is already declared there
  src/lca_setup.py    -> prepare_spatial_data(), build_demand()  (shared w/ 01)
  src/config.py       -> project name, method ids, output dir
Output (results/tables/):
  mc_scores_all_scenarios.csv    long format: scenario, category, iteration, score
  mc_summary_all_scenarios.csv   per (scenario, category): mean, median, sd,
                                 p2.5, p50, p97.5, CV%, n

USAGE
-----
  conda run -n bw25-regional python monte_carlo.py --test           # N=8 smoke test
  conda run -n bw25-regional python monte_carlo.py --iterations 1000 # full run
  conda run -n bw25-regional python monte_carlo.py -n 1000 --scenarios baseline,high_yield

RUNTIME
-------
~ (scenarios x 6 methods x N) solves. The four spatial methods are the slow
ones. Start with --test to time a few iterations before committing to a long
run; 1000 iterations across five scenarios is an overnight job.
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/ dir, so `src.` resolves

_prefix = sys.prefix
if os.name == "nt":
    os.environ.setdefault("GDAL_DATA", os.path.join(_prefix, "Library", "share", "gdal"))
    os.environ.setdefault("PROJ_LIB",  os.path.join(_prefix, "Library", "share", "proj"))

import numpy as np
import pandas as pd
import bw2data as bd
import bw2calc as bc
import bw2regional as bwr

from src.config import (
    PROJECT_NAME, RESULTS_TABLES_DIR,
    METHOD_LAND_USE_REGIONAL, METHOD_WATER, METHOD_N_EUTRO, METHOD_P_EUTRO,
    METHOD_CLIMATE_CHANGE, METHOD_ECOTOX_FW,
)
from src.scenarios import ALL_SCENARIOS
from src.inventory import build_foreground_db
from src.lca_setup import prepare_spatial_data, build_demand

# category label, method id, spatial? (spatial -> OneSpatialScaleLCA)
_METHODS = [
    ("Land use occupation", METHOD_LAND_USE_REGIONAL, True),
    ("Water consumption",   METHOD_WATER,             True),
    ("FW eutrophication N", METHOD_N_EUTRO,           True),
    ("FW eutrophication P", METHOD_P_EUTRO,           True),
    ("Climate change rcp26", METHOD_CLIMATE_CHANGE,   False),
    ("FW ecotoxicity",      METHOD_ECOTOX_FW,         False),
]


def _make_lca(demand, method, spatial, seed, stochastic=True):
    """Build an LCA object.

    stochastic=True gives the resampling object the Monte Carlo iterates.
    stochastic=False gives a plain deterministic solve — needed for the
    reference score, which previously ALSO passed use_distributions=True and
    so reported the first random draw in the 'deterministic' column. For
    ecotoxicity that draw can be negative (ecoinvent carries ~4,600 normal
    distributions narrow enough to sample below zero), which made the
    reference look nonsensical even though the true deterministic score is
    positive.
    """
    cls = bwr.OneSpatialScaleLCA if spatial else bc.LCA
    kwargs = dict(use_distributions=True, seed_override=seed) if stochastic else {}
    lca = cls(demand=demand, method=method, **kwargs)
    lca.lci()
    lca.lcia()
    return lca


def mc_scores(demand, method, spatial, n, seed):
    """Return a list of n Monte Carlo scores for one demand/method.

    next(lca) advances to the next joint sample of the distributions; we then
    recompute the characterised score. .lcia() after next() is required for the
    spatial classes and harmless (idempotent) for the standard LCA, so it is
    applied uniformly.
    """
    lca = _make_lca(demand, method, spatial, seed)
    out = []
    for _ in range(n):
        next(lca)
        lca.lcia()
        out.append(float(lca.score))
    return out


def run(scenarios, n, seed, out_tag=""):
    bd.projects.set_current(PROJECT_NAME)
    os.makedirs(RESULTS_TABLES_DIR, exist_ok=True)

    print(f"Preparing spatial data ...")
    spatial_data = prepare_spatial_data()

    long_rows, summary_rows = [], []
    t0 = time.time()

    for si, scen in enumerate(scenarios, 1):
        print(f"\n[{si}/{len(scenarios)}] {scen.name}: building foreground ...")
        build_foreground_db(scen, spatial_data)
        demand = build_demand(scen)

        for label, method, spatial in _METHODS:
            ts = time.time()
            # deterministic reference (no resampling at all)
            ref = _make_lca(demand, method, spatial, seed, stochastic=False)
            det = float(ref.score)

            draws = mc_scores(demand, method, spatial, n, seed)
            arr = np.asarray(draws, dtype=float)

            varied = np.unique(np.round(arr, 20)).size > 1
            dt = time.time() - ts
            print(f"    {label:<22} det={det:.3e}  "
                  f"MC median={np.median(arr):.3e}  CV={_cv(arr)*100:5.1f}%  "
                  f"vary={'Y' if varied else 'N'}  ({dt:.1f}s, {dt/max(n,1):.2f}s/it)")

            for it, val in enumerate(draws):
                long_rows.append({"scenario": scen.name, "category": label,
                                  "iteration": it, "score": val})
            summary_rows.append({
                "scenario": scen.name, "category": label,
                "deterministic": det,
                "mean": float(arr.mean()), "median": float(np.median(arr)),
                "sd": float(arr.std(ddof=1)) if n > 1 else 0.0,
                "p2.5":  float(np.percentile(arr, 2.5)),
                "p50":   float(np.percentile(arr, 50)),
                "p97.5": float(np.percentile(arr, 97.5)),
                "cv_pct": float(_cv(arr) * 100.0),
                "n": int(n), "varied": bool(varied),
            })

    long_df = pd.DataFrame(long_rows)
    summ_df = pd.DataFrame(summary_rows)
    lp = os.path.join(RESULTS_TABLES_DIR, f"mc_scores_all_scenarios{out_tag}.csv")
    sp = os.path.join(RESULTS_TABLES_DIR, f"mc_summary_all_scenarios{out_tag}.csv")
    long_df.to_csv(lp, index=False)
    summ_df.to_csv(sp, index=False)

    print(f"\nDone in {time.time()-t0:.0f}s.")
    print(f"Saved -> {lp}")
    print(f"Saved -> {sp}")

    n_novary = (~summ_df["varied"]).sum()
    if n_novary:
        print(f"\nWARNING: {n_novary} (scenario, category) combos showed NO variation "
              f"across draws — resampling may not be reaching those methods:")
        for _, r in summ_df[~summ_df["varied"]].iterrows():
            print(f"    {r['scenario']:<24} {r['category']}")
    return long_df, summ_df


def _cv(arr):
    m = arr.mean()
    return (arr.std(ddof=1) / m) if (len(arr) > 1 and m) else 0.0


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo uncertainty for the TZ cotton LCA")
    ap.add_argument("-n", "--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20240101)
    ap.add_argument("--scenarios", type=str, default="",
                    help="comma-separated scenario names; default = all")
    ap.add_argument("--test", action="store_true",
                    help="quick smoke test: 8 iterations, all scenarios/methods")
    args = ap.parse_args()

    n = 8 if args.test else args.iterations
    tag = "_test" if args.test else ""

    scens = ALL_SCENARIOS
    if args.scenarios:
        want = {s.strip() for s in args.scenarios.split(",")}
        scens = [s for s in ALL_SCENARIOS if s.name in want]
        if not scens:
            sys.exit(f"No scenarios matched {want}. "
                     f"Available: {[s.name for s in ALL_SCENARIOS]}")

    print(f"Monte Carlo: {len(scens)} scenario(s) x {len(_METHODS)} methods x {n} iterations, "
          f"seed={args.seed}")
    run(scens, n, args.seed, out_tag=tag)


if __name__ == "__main__":
    main()
