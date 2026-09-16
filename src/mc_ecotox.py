"""
mc_ecotox_stable.py
Recompute freshwater ecotoxicity Monte Carlo statistics on the NUMERICALLY
STABLE subset of characterised flows.

WHY
---
The full-inventory ecotoxicity MC has CV ~81% and ~10% negative draws. That is
not physical uncertainty. Diagnostics on the baseline showed:

  * 49 of 1,179 active characterised flows ever sample negative (4.2%), but
    they carry 74.3% of the total variance.
  * One flow — Cadmium II to agricultural soil — carries 67.7% of the variance
    on its own. Its DETERMINISTIC contribution is +2.80e-11 PDF*yr against a
    total score of 5.21e-06, i.e. 0.0005% of the result, yet its MC median is
    -2.59e-07 and it is negative in 53% of draws.
  * The underlying biosphere exchanges are all positive lognormals. The sign
    flips come from the technosphere solution: the emitting activities are
    waste treatments, which carry negative scaling factors under ecoinvent's
    cutoff convention, and cadmium's very high CF amplifies small swings.

Excluding the sign-crossing flows gives CV ~49% with no negative draws — still
the widest of the six categories, which is the substantive finding.

WHAT IT DOES
------------
Re-runs ONLY the ecotoxicity method (non-spatial, so fast), recording per-flow
characterised contributions for every draw. Within each scenario it flags flows
whose draws change sign, then reports statistics both with and without them.

Uses the same seed as monte_carlo.py, so the "full" column here should
reproduce the ecotoxicity rows of mc_summary_all_scenarios.csv.

USAGE
-----
  conda run -n bw25-regional python mc_ecotox_stable.py            # N=1000
  conda run -n bw25-regional python mc_ecotox_stable.py -n 50      # quick check

OUTPUT (results/tables/)
  mc_ecotox_stable.csv         per-scenario full vs stable statistics
  mc_ecotox_stable_draws.csv   per-draw stable scores (for the whisker plot)
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/ dir, so `src.` resolves

import numpy as np
import pandas as pd
import bw2data as bd
import bw2calc as bc

from src.config import PROJECT_NAME, RESULTS_TABLES_DIR, METHOD_ECOTOX_FW
from src.scenarios import ALL_SCENARIOS
from src.inventory import build_foreground_db
from src.lca_setup import prepare_spatial_data, build_demand

# A flow counts as sign-crossing if it samples negative in more than this
# fraction of draws.
#
# This was originally 0.001 — "goes negative even once in 1000 draws". At
# N = 1000 that is far too aggressive: it catches flows that are merely WIDE,
# not genuinely sign-ambiguous, and the excluded set then carried 10.5% of the
# deterministic score. A threshold sweep on baseline (N = 400) shows a sharp
# cliff between the two settings, and almost no variance cost to the looser one:
#
#   threshold   flows cut   % det cut    CV%    var cut%
#      0.0005          49       10.38   45.9        76.1
#      0.0020          49       10.38   45.9        76.1
#      0.0100          43        1.25   41.7        75.6
#      0.0500          42        1.25   41.7        75.6
#      0.3000          37        1.24   41.7        75.6
#
# Six flows separate 10.38% from 1.25% of the deterministic score while
# contributing 0.5 percentage points of variance between them. Excluding a
# tenth of the result to remove half a point of variance is not defensible;
# excluding 1.25% to remove 75.6% of it is.
NEG_RATE_THRESHOLD = 0.01


def run(scenarios, n, seed, resume=True):
    bd.projects.set_current(PROJECT_NAME)
    os.makedirs(RESULTS_TABLES_DIR, exist_ok=True)
    spatial_data = prepare_spatial_data()

    sp = os.path.join(RESULTS_TABLES_DIR, "mc_ecotox_stable.csv")
    dp = os.path.join(RESULTS_TABLES_DIR, "mc_ecotox_stable_draws.csv")

    # Results are appended per scenario rather than written once at the end.
    # The first attempt at this run died partway through scenario 3 (exit 120,
    # no traceback) and lost the two completed scenarios with it; on a ~100 min
    # job over a network drive that has dropped repeatedly, a crash should cost
    # one scenario, not all six.
    done = set()
    if resume and os.path.exists(sp):
        try:
            done = set(pd.read_csv(sp)["scenario"])
            print(f"resuming; already complete: {sorted(done)}")
        except Exception:
            done = set()

    rows, draw_rows = [], []
    t0 = time.time()

    for si, scen in enumerate(scenarios, 1):
        if scen.name in done:
            print(f"\n[{si}/{len(scenarios)}] {scen.name}: already done, skipping")
            continue
        print(f"\n[{si}/{len(scenarios)}] {scen.name}: building foreground ...")
        build_foreground_db(scen, spatial_data)
        demand = build_demand(scen)

        det_lca = bc.LCA(demand, METHOD_ECOTOX_FW)
        det_lca.lci(); det_lca.lcia()
        det = float(det_lca.score)

        lca = bc.LCA(demand, METHOD_ECOTOX_FW, use_distributions=True,
                     seed_override=seed)
        lca.lci(); lca.lcia()

        nflow = lca.characterized_inventory.shape[0]
        M = np.zeros((n, nflow), dtype=float)
        ts = time.time()
        for k in range(n):
            next(lca)
            lca.lcia()
            M[k, :] = np.asarray(lca.characterized_inventory.sum(axis=1)).ravel()

        full = M.sum(axis=1)
        active = np.where(np.abs(M).sum(axis=0) > 0)[0]
        neg_rate = (M[:, active] < 0).mean(axis=0)
        crossing = active[neg_rate > NEG_RATE_THRESHOLD]
        stable_cols = np.setdiff1d(active, crossing)
        stable = M[:, stable_cols].sum(axis=1)

        # deterministic score restricted to the stable flows
        det_all = np.asarray(det_lca.characterized_inventory.sum(axis=1)).ravel()
        det_stable = float(det_all[stable_cols].sum())

        dt = time.time() - ts

        def stats(a):
            return dict(mean=float(a.mean()), median=float(np.median(a)),
                        sd=float(a.std(ddof=1)),
                        p2_5=float(np.percentile(a, 2.5)),
                        p97_5=float(np.percentile(a, 97.5)),
                        cv_pct=float(100 * a.std(ddof=1) / a.mean()) if a.mean() else np.nan,
                        neg_pct=float(100 * (a < 0).mean()))

        sf, ss = stats(full), stats(stable)
        rows.append({
            "scenario": scen.name, "n": n,
            "deterministic_full": det, "deterministic_stable": det_stable,
            "n_active_flows": len(active), "n_excluded_flows": len(crossing),
            "neg_rate_threshold": NEG_RATE_THRESHOLD,
            "excluded_share_of_det_pct":
                float(100 * det_all[crossing].sum() / det) if det else np.nan,
            **{f"full_{k}": v for k, v in sf.items()},
            **{f"stable_{k}": v for k, v in ss.items()},
        })
        for k in range(n):
            draw_rows.append({"scenario": scen.name, "iteration": k,
                              "score_full": full[k], "score_stable": stable[k]})

        print(f"    det={det:.3e}  full: median={sf['median']:.3e} CV={sf['cv_pct']:.1f}% "
              f"neg={sf['neg_pct']:.1f}%")
        print(f"    {'':16s}stable: median={ss['median']:.3e} CV={ss['cv_pct']:.1f}% "
              f"neg={ss['neg_pct']:.1f}%   ({len(crossing)} of {len(active)} flows "
              f"excluded, {rows[-1]['excluded_share_of_det_pct']:.4f}% of det score)")
        print(f"    ({dt:.0f}s, {dt/max(n,1):.2f}s/it)")
        del M

        pd.DataFrame(rows[-1:]).to_csv(
            sp, mode="a", header=not os.path.exists(sp), index=False)
        pd.DataFrame(draw_rows[-n:]).to_csv(
            dp, mode="a", header=not os.path.exists(dp), index=False)
        print(f"    appended -> mc_ecotox_stable.csv / _draws.csv")

    print(f"\nDone in {time.time()-t0:.0f}s.")
    summ = pd.read_csv(sp) if os.path.exists(sp) else pd.DataFrame()
    print(f"Saved -> {sp}  ({len(summ)} scenarios)")
    print(f"Saved -> {dp}")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20240101)
    ap.add_argument("--scenarios", type=str, default="",
                    help="comma-separated scenario names; default = all")
    ap.add_argument("--fresh", action="store_true",
                    help="delete any existing output and start over")
    args = ap.parse_args()

    scens = ALL_SCENARIOS
    if args.scenarios:
        want = {x.strip() for x in args.scenarios.split(",")}
        scens = [x for x in ALL_SCENARIOS if x.name in want]
        if not scens:
            sys.exit(f"no scenarios matched {want}")
    if args.fresh:
        for f in ("mc_ecotox_stable.csv", "mc_ecotox_stable_draws.csv"):
            fp = os.path.join(RESULTS_TABLES_DIR, f)
            if os.path.exists(fp):
                os.remove(fp)
                print(f"removed {f}")

    print(f"Ecotoxicity stable-subset MC: {len(scens)} scenarios x "
          f"{args.iterations} iterations, seed={args.seed}")
    run(scens, args.iterations, args.seed, resume=not args.fresh)


if __name__ == "__main__":
    main()
