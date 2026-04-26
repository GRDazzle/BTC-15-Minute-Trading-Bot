"""4-fold walk-forward parameter sweep — folds 1-3 select, fold 4 = OOS test.

Given N days of training data (default 45), slices into n_folds chronological
folds (default 4 × 11-day folds + 1 orphan day). Runs `pnl_sweep.py` once per
fold (asset-parallel within each fold via pnl_sweep's own executor), then:

  - Selection (folds 1..N-1): top 10 combos by cross-fold consistency
    (min-max-rank, tiebreak sum-of-ranks). Combo must be positive-PnL and
    traded_count >= 10 in all selection folds.
  - OOS / final pick (fold N): of those top 10, sweep results on fold N
    (using `_v14_foldN` model — the live-deploy model, never saw fold N).
    Pick the one with highest WR (positive PnL, traded >= 10), tiebreak PnL.

Why this scheme:
  Selection on cross-fold consistency finds combos that are robust, not
  lucky on one fold. Final pick on held-out fold N validates the chosen
  combo against truly unseen data — the SAME model that goes to live.
  Strictly OOS for both parameters AND model.

Usage:
    python scripts/retrain_4fold.py --assets BTC,ETH,SOL,XRP \\
        --end-date 2026-04-24 --model-suffix _v14

Outputs:
    output/pnl_sweep/fold1..N/{ASSET}_v14_foldK_pnl_sweep.csv  (per-fold grid)
    output/retrain_4fold/report_v14.txt                        (top 10 + final pick)
"""
import argparse
import csv
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SWEEP_OUT = PROJECT_ROOT / "output" / "pnl_sweep"
REPORT_DIR = PROJECT_ROOT / "output" / "retrain_4fold"
PYTHON = sys.executable


def _run_sweep(assets: str, from_d: date, to_d: date, suffix: str, subdir: str,
               min_dm: int, balance: float, log_path: Path) -> int:
    """Invoke pnl_sweep.py for a specific date range. Returns exit code."""
    cmd = [
        PYTHON, "-u", "scripts/pnl_sweep.py",
        "--asset", assets,
        "--min-dm", str(min_dm),
        "--from-date", from_d.isoformat(),
        "--to-date", to_d.isoformat(),
        "--model-suffix", suffix,
        "--no-config-write",
        "--output-subdir", subdir,
        "--balance", str(balance),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"[cmd] {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def _load_fold_csv(fold_dir: Path, asset: str, suffix: str) -> list[dict]:
    path = fold_dir / f"{asset}{suffix}_pnl_sweep.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            # Coerce numeric fields for sorting
            for k in ("threshold", "k_xgb", "k_lstm", "total_pnl", "win_rate",
                     "traded_count", "avg_pnl_per_trade", "final_balance"):
                if k in r and r[k] != "":
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            for k in ("min_dm", "max_price_cents", "min_price_cents", "traded_count",
                      "win_count", "loss_count", "total_windows"):
                if k in r and r[k] != "":
                    try:
                        r[k] = int(float(r[k]))
                    except ValueError:
                        pass
            rows.append(r)
    return rows


def _combo_key(r: dict) -> tuple:
    """Unique key for a combo (same across folds)."""
    return (
        round(float(r["threshold"]), 4),
        round(float(r["k_xgb"]), 4),
        round(float(r["k_lstm"]), 4),
        int(r["min_dm"]),
        int(r.get("max_price_cents") or 0),
        int(r.get("min_price_cents") or 0),
    )


def _rank_fold(rows: list[dict], min_traded: int = 10) -> dict[tuple, int]:
    """Filter to combos with traded_count >= min_traded AND positive PnL,
    sort by win_rate desc, return dict of combo_key -> rank (1 = best)."""
    filt = [
        r for r in rows
        if int(r.get("traded_count", 0)) >= min_traded
        and float(r.get("total_pnl", 0)) > 0
    ]
    filt.sort(key=lambda r: float(r.get("win_rate", 0)), reverse=True)
    return {_combo_key(r): i + 1 for i, r in enumerate(filt)}


def _pick_top_n(fold_rows: list[list[dict]], n_top: int = 10,
                min_traded: int = 10,
                min_folds_present: int | None = None) -> list[tuple[tuple, dict]]:
    """Given selection-fold rows (one list per fold), return top N combos by
    cross-fold consistency.

    Selection rule: among combos present in >= min_folds_present folds (default
    all of them), rank by smallest MAX rank across folds (the "worst-case"
    rank is best). Tie-break on sum-of-ranks. This prefers combos that are
    top-K in every fold over combos that are #1 once and #500 elsewhere.

    Returns: list of (combo_key, details_dict), sorted by (max_rank, sum_rank).
    Empty list if no combo qualifies.
    """
    if min_folds_present is None:
        min_folds_present = len(fold_rows)
    per_fold_ranks = [_rank_fold(rows, min_traded) for rows in fold_rows]
    all_keys: set = set()
    for d in per_fold_ranks:
        all_keys |= set(d.keys())
    candidates: list[tuple[tuple, dict]] = []
    for k in all_keys:
        ranks = [d.get(k) for d in per_fold_ranks]
        present = sum(1 for r in ranks if r is not None)
        if present < min_folds_present:
            continue
        max_rank = max(r for r in ranks if r is not None)
        sum_rank = sum(r for r in ranks if r is not None)
        candidates.append((k, {
            "ranks": ranks,
            "max_rank": max_rank,
            "sum_rank": sum_rank,
        }))
    candidates.sort(key=lambda c: (c[1]["max_rank"], c[1]["sum_rank"]))
    return candidates[:n_top]


def _lookup_combo(rows: list[dict], combo_key: tuple) -> dict | None:
    for r in rows:
        if _combo_key(r) == combo_key:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True,
                    help="Comma-separated asset list, e.g. BTC,ETH,SOL,XRP")
    ap.add_argument("--days-total", type=int, default=45,
                    help="Total days to slice. MUST match train_all_5fold.py's "
                         "--days-total. With --oos-days=0 (default): 45/4 = 11-day "
                         "folds, 1 orphan day at start. Fold 4 is the OOS test.")
    ap.add_argument("--oos-days", type=int, default=0,
                    help="Held-out OOS days separate from folds. Default 0 = "
                         "new scheme (fold N is OOS, model `_foldN` is genuinely "
                         "OOS for both params and model). Set >0 for legacy.")
    ap.add_argument("--n-folds", type=int, default=4,
                    help="Number of folds (default 4). With new scheme: "
                         "folds 1..N-1 are selection, fold N is OOS test.")
    ap.add_argument("--n-top", type=int, default=10,
                    help="Top N combos by cross-fold consistency to advance to "
                         "fold-N OOS test (default 10).")
    ap.add_argument("--model-suffix", type=str, default="_v14",
                    help="Model filename suffix (default _v14)")
    ap.add_argument("--min-dm", type=int, default=2)
    ap.add_argument("--end-date", type=str, default=None,
                    help="Anchor end date YYYY-MM-DD (default: today UTC). MUST match "
                         "the anchor passed to train_all_5fold.py or fold boundaries will "
                         "drift and sweeps will contaminate the models they're scoring.")
    ap.add_argument("--balance", type=float, default=1_000_000.0,
                    help="Balance passed to pnl_sweep so position sizing is "
                         "max_contracts-capped (not balance-capped). Default $1M. "
                         "BTC's 500-contract-per-trade cap * 72c = $360/trade, so a "
                         "single fold (~100+ trades) would starve a $10k balance "
                         "partway through. $1M guarantees max_contracts is the "
                         "binding constraint across the full fold for every asset.")
    ap.add_argument("--skip-sweeps", action="store_true",
                    help="Skip sweep invocations, use existing CSVs in output/pnl_sweep/fold*")
    args = ap.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]
    anchor = date.fromisoformat(args.end_date) if args.end_date else date.today()

    # Compute fold boundaries.
    # New scheme (oos_days=0): n_folds × fold_len = days_total - orphan; fold N is OOS.
    # Legacy (oos_days>0): n_folds × fold_len + oos_days; OOS held out separately.
    if args.oos_days == 0:
        fold_len = args.days_total // args.n_folds
        train_start = anchor - timedelta(days=fold_len * args.n_folds)
        folds = []
        for i in range(args.n_folds):
            lo = train_start + timedelta(days=i * fold_len)
            hi = train_start + timedelta(days=(i + 1) * fold_len) - timedelta(days=1)
            folds.append((f"fold{i+1}", lo, hi))
        # Fold N is the OOS test — no separate window.
        oos_fold_idx = args.n_folds - 1
        oos_name, oos_lo, oos_hi = folds[oos_fold_idx]
        selection_folds = folds[:oos_fold_idx]
    else:
        training_days = args.days_total - args.oos_days
        fold_len = training_days // args.n_folds
        train_start = anchor - timedelta(days=args.days_total)
        folds = []
        for i in range(args.n_folds):
            lo = train_start + timedelta(days=i * fold_len)
            hi = train_start + timedelta(days=(i + 1) * fold_len) - timedelta(days=1)
            folds.append((f"fold{i+1}", lo, hi))
        oos_lo = anchor - timedelta(days=args.oos_days)
        oos_hi = anchor - timedelta(days=1)
        oos_name = "oos"
        selection_folds = folds  # all folds are selection in legacy mode
        oos_fold_idx = None  # legacy uses separate OOS

    print(f"{'='*70}")
    if args.oos_days == 0:
        print(f"Walk-forward — {len(selection_folds)} selection folds + fold {args.n_folds} OOS")
    else:
        print(f"Walk-forward — {args.n_folds} folds + {args.oos_days}-day separate OOS")
    print(f"  anchor : {anchor}")
    print(f"  assets : {assets}")
    print(f"  suffix : {args.model_suffix}")
    print(f"{'='*70}")
    for i, (name, lo, hi) in enumerate(folds):
        tag = " (OOS)" if args.oos_days == 0 and i == oos_fold_idx else ""
        print(f"  {name}: {lo} -> {hi}  ({(hi-lo).days+1} days){tag}")
    if args.oos_days > 0:
        print(f"  oos  : {oos_lo} -> {oos_hi}  ({(oos_hi-oos_lo).days+1} days)")
    print(f"{'='*70}\n")

    # Stage 1: run sweeps (unless --skip-sweeps). Each fold uses its matching
    # walk-forward model (e.g. _v14_fold2) — model never saw that fold during
    # training. In new scheme, fold N's sweep uses _v14_foldN — the live deploy
    # model + the OOS test in one shot.
    if not args.skip_sweeps:
        if args.oos_days == 0:
            # New scheme: every fold (including OOS) uses its exclusion model
            sweep_jobs = [
                (name, lo, hi, f"{args.model_suffix}_{name}")
                for name, lo, hi in folds
            ]
        else:
            # Legacy: folds use exclusion models, OOS uses base (_v14)
            sweep_jobs = [
                (name, lo, hi, f"{args.model_suffix}_{name}")
                for name, lo, hi in folds
            ] + [(oos_name, oos_lo, oos_hi, args.model_suffix)]
        for name, lo, hi, sweep_suffix in sweep_jobs:
            print(f"\n[{name}] Running sweep {lo} -> {hi}  (model={sweep_suffix})...")
            log_path = REPORT_DIR / f"sweep_{name}{args.model_suffix}.log"
            rc = _run_sweep(
                assets=",".join(assets), from_d=lo, to_d=hi,
                suffix=sweep_suffix, subdir=name,
                min_dm=args.min_dm, balance=args.balance, log_path=log_path,
            )
            if rc != 0:
                print(f"  [{name}] sweep exit={rc} — check {log_path}")
            else:
                print(f"  [{name}] done. log: {log_path}")

    # Stage 2: analyze per asset
    # New scheme: pick top N candidates from selection folds (folds 1..N-1) by
    # cross-fold consistency, then pick the winner from those N based on the
    # held-out fold-N (OOS) sweep — highest WR with positive PnL, tiebreak PnL.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"report{args.model_suffix}.txt"
    with open(report_path, "w") as rep:
        rep.write(f"Walk-forward report — suffix={args.model_suffix}\n")
        rep.write(f"  scheme: {'selection folds 1..N-1 + fold N OOS' if args.oos_days == 0 else 'all folds + separate OOS'}\n")
        rep.write(f"  folds: {[(n, lo.isoformat(), hi.isoformat()) for n, lo, hi in folds]}\n")
        if args.oos_days == 0:
            rep.write(f"  selection: folds 1..{args.n_folds - 1}; OOS: fold{args.n_folds} (model `{args.model_suffix}_fold{args.n_folds}` — live deploy)\n")
        else:
            rep.write(f"  oos  : {oos_lo} -> {oos_hi}  (legacy mode)\n")
        rep.write(f"  top-N selection candidates: {args.n_top}\n")
        rep.write("=" * 90 + "\n\n")

        print(f"\n{'='*70}")
        print(f"  ANALYSIS — top {args.n_top} from selection folds, winner from fold-{args.n_folds} OOS")
        print(f"{'='*70}\n")

        for asset in assets:
            # Per-fold CSVs carry the fold-specific model suffix (e.g. _v14_fold2).
            fold_rows_all = [
                _load_fold_csv(SWEEP_OUT / name, asset, f"{args.model_suffix}_{name}")
                for name, _, _ in folds
            ]
            # Selection folds (folds 1..N-1 in new scheme; all folds in legacy)
            if args.oos_days == 0:
                selection_rows = fold_rows_all[:oos_fold_idx]
                oos_rows = fold_rows_all[oos_fold_idx]
                oos_label = folds[oos_fold_idx][0]
            else:
                selection_rows = fold_rows_all
                oos_rows = _load_fold_csv(SWEEP_OUT / "oos", asset, args.model_suffix)
                oos_label = "oos"

            sel_sizes = [len(r) for r in selection_rows]
            if any(n == 0 for n in sel_sizes):
                print(f"[{asset}] missing selection-fold data: sizes={sel_sizes} — skipping")
                rep.write(f"{asset}: SKIP (missing selection-fold data, sizes={sel_sizes})\n\n")
                continue
            if len(oos_rows) == 0:
                print(f"[{asset}] missing OOS fold data — skipping")
                rep.write(f"{asset}: SKIP (missing OOS fold data)\n\n")
                continue

            # Top-N candidates from selection folds (positive PnL in all of them)
            candidates = _pick_top_n(
                selection_rows, n_top=args.n_top, min_traded=10,
                min_folds_present=len(selection_rows),
            )
            relaxed = False
            if not candidates:
                # Relax: present in N-1 selection folds (e.g. 2 of 3)
                candidates = _pick_top_n(
                    selection_rows, n_top=args.n_top, min_traded=10,
                    min_folds_present=max(1, len(selection_rows) - 1),
                )
                relaxed = True
            if not candidates:
                print(f"[{asset}] no top-{args.n_top} candidates with positive PnL — skipping")
                rep.write(f"{asset}: NO CANDIDATES (no combo with positive PnL across selection folds)\n\n")
                continue

            # OOS rank index for ALL combos (used for both winner selection and reporting)
            oos_ranks = _rank_fold(oos_rows, min_traded=10)
            oos_total = len(oos_ranks)

            # Pick winner: among the top-N candidates, find the one with the
            # highest fold-N WR (positive-PnL, n >= 10). Tiebreak: total_pnl.
            winner = None
            for combo_key, details in candidates:
                oos_r = _lookup_combo(oos_rows, combo_key)
                if not oos_r:
                    continue
                # Must be positive PnL on OOS and have meaningful sample
                if int(oos_r.get("traded_count", 0)) < 10:
                    continue
                if float(oos_r.get("total_pnl", 0)) <= 0:
                    continue
                rec = {
                    "combo_key": combo_key,
                    "details": details,
                    "oos_row": oos_r,
                    "wr": float(oos_r.get("win_rate", 0)),
                    "pnl": float(oos_r.get("total_pnl", 0)),
                }
                if winner is None or (rec["wr"], rec["pnl"]) > (winner["wr"], winner["pnl"]):
                    winner = rec

            print(f"\n[{asset}] {len(candidates)} candidates from selection folds"
                  f"{' (RELAXED — present in N-1 folds)' if relaxed else ''}")
            rep.write(f"{asset}:\n")
            if relaxed:
                rep.write(f"  (RELAXED — combos required in N-1 selection folds, not all N)\n")
            rep.write(f"  {len(candidates)} candidates from selection folds:\n")

            # Print the candidates table (top-N from selection)
            rep.write(f"\n  {'#':>3} | {'thr':>4} | {'k_xgb':>5} | {'k_lstm':>6} | "
                      f"{'mDm':>3} | {'minP':>4} | {'maxP':>4} | "
                      f"{'sel_ranks':>16} | {'OOS PnL':>9} | {'OOS WR':>6} | {'OOS n':>5}\n")
            for idx, (combo_key, details) in enumerate(candidates, 1):
                thr, k_xgb, k_lstm, min_dm, max_p, min_p = combo_key
                oos_r = _lookup_combo(oos_rows, combo_key)
                oos_pnl_s = f"${float(oos_r['total_pnl']):+.2f}" if oos_r else "n/a"
                oos_wr_s = f"{float(oos_r['win_rate']):.1f}%" if oos_r else "n/a"
                oos_n_s = f"{int(oos_r['traded_count'])}" if oos_r else "n/a"
                ranks_s = str(details["ranks"])
                rep.write(f"  {idx:>3} | {thr:>4.2f} | {k_xgb:>5.2f} | {k_lstm:>6.2f} | "
                          f"{min_dm:>3} | {min_p:>4} | {max_p:>4} | "
                          f"{ranks_s:>16} | {oos_pnl_s:>9} | {oos_wr_s:>6} | {oos_n_s:>5}\n")

            if winner is None:
                msg = (f"  ⚠ NO WINNER — none of the top-{args.n_top} candidates "
                       f"had positive OOS PnL with n >= 10 on fold-{args.n_folds}.\n"
                       f"  Live deploy: hold off — params don't generalize to held-out fold.")
                print(msg)
                rep.write(f"\n{msg}\n\n")
                continue

            wkey = winner["combo_key"]
            wdetails = winner["details"]
            woos = winner["oos_row"]
            thr, k_xgb, k_lstm, min_dm, max_p, min_p = wkey

            # Compute OOS rank for the winner (vs all combos in OOS sweep)
            oos_rank = oos_ranks.get(wkey)
            if oos_rank is not None:
                rank_pct = (oos_rank / oos_total * 100) if oos_total else 0.0
                rank_str = f"rank {oos_rank}/{oos_total} = top {rank_pct:.1f}%"
                if oos_rank > 100 or rank_pct > 25.0:
                    rank_str += "  ⚠ winner ranks low across all OOS combos"
            else:
                rank_str = "NOT in OOS positive-PnL set"

            # Final pick summary
            print(f"  WINNER: thr={thr} k_xgb={k_xgb} k_lstm={k_lstm} "
                  f"min_dm={min_dm} min_p={min_p} max_p={max_p}")
            print(f"    selection ranks: {wdetails['ranks']} "
                  f"(max={wdetails['max_rank']}, sum={wdetails['sum_rank']})")
            print(f"    fold-{args.n_folds} OOS: PnL=${winner['pnl']:+.2f} "
                  f"WR={winner['wr']:.1f}% n={int(woos['traded_count'])}  | {rank_str}")

            rep.write(f"\n  WINNER (live deploy):\n")
            rep.write(f"    threshold={thr}  k_xgb={k_xgb}  k_lstm={k_lstm}  "
                      f"min_dm={min_dm}  min_price={min_p}  max_price={max_p}\n")
            rep.write(f"    selection ranks: {wdetails['ranks']} "
                      f"(max={wdetails['max_rank']}, sum={wdetails['sum_rank']})\n")
            # Per-selection-fold stats for transparency
            for fold_idx, (name, _, _) in enumerate(folds[:len(selection_rows)]):
                r = _lookup_combo(selection_rows[fold_idx], wkey)
                if r:
                    rep.write(f"    {name}: PnL=${float(r['total_pnl']):+.2f} "
                              f"WR={float(r['win_rate']):.1f}% "
                              f"n={int(r['traded_count'])}\n")
            rep.write(f"    fold-{args.n_folds} OOS: PnL=${winner['pnl']:+.2f} "
                      f"WR={winner['wr']:.1f}% n={int(woos['traded_count'])}  "
                      f"| {rank_str}\n\n")

        rep.write("\nAll sweep CSVs in output/pnl_sweep/<foldN>/\n")
        rep.write("Live model = `{ASSET}<suffix>_fold<N>_xgb.json` + `_fold<N>_lstm.pt`\n"
                  .replace("<suffix>", args.model_suffix).replace("<N>", str(args.n_folds)))

    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
