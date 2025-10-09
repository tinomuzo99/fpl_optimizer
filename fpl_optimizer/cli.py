from __future__ import annotations
import argparse
import os
import pandas as pd
from tabulate import tabulate

from .config import REQUIRED_COUNTS, MAX_PER_TEAM, HORIZON_GWS
from .name_matching import map_names_to_pool
from .optimizer import (
    prepare_player_pool, optimise_squad, choose_starting_xi, best_single_transfer
)

def parse_args():
    p = argparse.ArgumentParser(prog="fpl", description="FPL optimiser CLI (saves CSV outputs)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_opts(sp):
        sp.add_argument("--horizon", type=int, default=HORIZON_GWS, help="Number of future gameweeks for horizon weighting")
        sp.add_argument("--min-play-chance", type=int, default=0, help="Min chance_of_playing_next_round when allow_flagged is False")
        sp.add_argument("--allow-flagged", action="store_true", help="Include flagged players in pool (penalised by availability multipliers)")
        sp.add_argument("--budget", type=float, default=100.0, help="Budget in millions (for optimise)")
        sp.add_argument("--lock", nargs="*", default=[], help="Force-include names (substring match)")
        sp.add_argument("--ban", nargs="*", default=[], help="Force-exclude names (substring match)")
        sp.add_argument("--out-dir", default="outputs", help="Directory to write CSV outputs")
        sp.add_argument("--score-type", choices=["next", "horizon"], default="horizon",
                        help="Which expected points to use: 'next' (1 GW) or 'horizon' (multi-GW)")

    p_opt = sub.add_parser("optimise", help="Optimise a fresh 15-man squad")
    common_opts(p_opt)

    p_xi = sub.add_parser("xi", help="Choose best starting XI for a given 15-man squad (CSV input)")
    common_opts(p_xi)
    p_xi.add_argument("--squad-file", required=True, help="Path to a CSV with 'name' column (optional 'team' column)")

    p_tr = sub.add_parser("transfers", help="Suggest best single transfer for current squad (CSV input)")
    common_opts(p_tr)
    p_tr.add_argument("--squad-file", required=True, help="Path to a CSV with 'name' column (optional 'team' column)")
    p_tr.add_argument("--bank", type=float, default=0.0, help="Money in the bank (millions)")
    p_tr.add_argument("--top-k", type=int, default=10, help="Top K transfer suggestions")

    return p.parse_args()

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def print_table(df: pd.DataFrame, headers="keys", floatfmt=".2f"):
    if df.empty:
        print("(no results)")
        return
    print(tabulate(df, headers=headers, showindex=False, tablefmt="github", floatfmt=floatfmt))

def read_squad_csv(path: str) -> pd.DataFrame:
    """
    Read a squad file expected to be CSV with:
      - required column: name
      - optional column: team
    Returns a DataFrame with columns ['name'] or ['name','team'].
    """
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    if "name" not in cols_lower:
        raise ValueError("Input squad CSV must contain a 'name' column.")
    name_col = cols_lower["name"]
    out = pd.DataFrame({"name": df[name_col].astype(str).str.strip()})
    # preserve optional team column if present
    if "team" in cols_lower:
        team_col = cols_lower["team"]
        out["team"] = df[team_col].astype(str).str.strip()
        # treat blank team as missing
        out.loc[out["team"].eq("") | out["team"].str.lower().eq("nan"), "team"] = None
    return out

def run_optimise(args):
    ensure_outdir(args.out_dir)
    pool_df = prepare_player_pool(
        allow_flagged=args.allow_flagged,
        min_play_chance=args.min_play_chance,
        horizon=args.horizon
    )
    tag, selected, total_cost, total_points, score_col = optimise_squad(
        pool_df,
        budget_m=args.budget,
        score_type=args.score_type,
        max_per_team=MAX_PER_TEAM,
        required_counts=REQUIRED_COUNTS,
        lock_names=set(args.lock),
        ban_names=set(args.ban),
        min_from_team=None,
        verbose=False,
    )

    # Save the optimised 15-man squad with BOTH metrics for transparency
    show = selected.sort_values(["pos_code","team_name","name"]).copy()
    cols = ["name","team_name","pos_code","cost_m","status","avail_mult",
            "exp_points_next",
            "exp_points_h"]
    out_csv = os.path.join(args.out_dir, f"optimised_squad_{args.score_type}.csv")
    show[cols].to_csv(out_csv, index=False)

    # Console view
    pshow = show.copy()
    pshow["cost_m"] = pshow["cost_m"].map(lambda x: f"£{x:0.1f}m")
    pshow[score_col] = pshow[score_col].map(lambda x: f"{x:0.2f}")
    print(f"\n[info] Strategy used: {tag} | Scoring: {args.score_type} ({score_col})")
    print_table(pshow[["name","team_name","pos_code","cost_m",score_col]])

    print("\nTotals over chosen metric:")
    print(f"  Cost: £{total_cost:0.1f}m / {args.budget:0.1f}m")
    print(f"  Sum({args.score_type}): {total_points:0.2f}")
    print(f"[saved] {out_csv}")

    # XI + Bench (using same metric)
    xi, bench, xi_pts = choose_starting_xi(selected, args.score_type)
    xi_out = os.path.join(args.out_dir, f"starting_xi_{args.score_type}.csv")
    bench_out = os.path.join(args.out_dir, f"bench_{args.score_type}.csv")
    xi.sort_values(['pos_code','team_name','name']).to_csv(xi_out, index=False)
    bench.sort_values(['pos_code','team_name','name']).to_csv(bench_out, index=False)

    xishow = xi.sort_values(['pos_code','team_name','name']).copy()
    xishow[score_col] = xishow[score_col].map(lambda x: f"{x:0.2f}")
    print("\nStarting XI:")
    print_table(xishow[["name","team_name","pos_code",score_col]])
    print(f"XI expected points ({args.score_type}): {xi_pts:0.2f}")
    print(f"[saved] {xi_out}\n[saved] {bench_out}")

def run_xi(args):
    ensure_outdir(args.out_dir)
    names_df = read_squad_csv(args.squad_file)                # <-- pass DataFrame
    pool_df = prepare_player_pool(
        allow_flagged=args.allow_flagged,
        min_play_chance=args.min_play_chance,
        horizon=args.horizon
    )
    current_df = map_names_to_pool(pool_df, names_df)         # <-- DataFrame, not list
    xi, bench, xi_pts = choose_starting_xi(current_df, args.score_type)

    score_col = "exp_points_next_avail" if args.score_type == "next" else "exp_points_h_avail"
    xishow = xi.sort_values(['pos_code','team_name','name']).copy()
    xishow[score_col] = xishow[score_col].map(lambda x: f"{x:0.2f}")

    print(f"\nBest XI using '{args.score_type}' metric (col: {score_col}): {xi_pts:.2f}")
    print_table(xishow[["name","team_name","pos_code",score_col]])

    xi_csv = os.path.join(args.out_dir, f"xi_{args.score_type}.csv")
    bench_csv = os.path.join(args.out_dir, f"bench_from_input_{args.score_type}.csv")
    xi.sort_values(['pos_code','team_name','name']).to_csv(xi_csv, index=False)
    bench.sort_values(['pos_code','team_name','name']).to_csv(bench_csv, index=False)
    print(f"[saved] {xi_csv}\n[saved] {bench_csv}")

def run_transfers(args):
    ensure_outdir(args.out_dir)
    names_df = read_squad_csv(args.squad_file)                # <-- pass DataFrame
    pool_df = prepare_player_pool(
        allow_flagged=args.allow_flagged,
        min_play_chance=args.min_play_chance,
        horizon=args.horizon
    )
    current_df = map_names_to_pool(pool_df, names_df)         # <-- DataFrame, not list

    recs = best_single_transfer(
        pool_df,
        current_df,
        bank_m=args.bank,
        top_k=args.top_k,
        score_type=args.score_type,
    )

    out_csv = os.path.join(args.out_dir, f"transfer_recommendations_{args.score_type}.csv")
    recs.to_csv(out_csv, index=False)

    if recs.empty:
        print("(no beneficial single-transfer found under constraints)")
    else:
        rshow = recs.copy()
        rshow["out_cost"] = rshow["out_cost"].map(lambda x: f"£{x:0.1f}m")
        rshow["in_cost"]  = rshow["in_cost"].map(lambda x: f"£{x:0.1f}m")
        rshow["delta_pts"] = rshow["delta_pts"].map(lambda x: f"{x:0.2f}")
        print_table(rshow, headers="keys")
    print(f"[saved] {out_csv}")

def main():
    args = parse_args()
    if args.cmd == "optimise":
        run_optimise(args)
    elif args.cmd == "xi":
        run_xi(args)
    elif args.cmd == "transfers":
        run_transfers(args)
    else:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
