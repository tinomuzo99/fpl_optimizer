from __future__ import annotations
import argparse
import os
import pandas as pd
from tabulate import tabulate

from .config import REQUIRED_COUNTS, MAX_PER_TEAM, HORIZON_GWS
from .name_matching import map_names_to_pool

from .tracking import (
    record_recommendation,
    settle_finished_recommendations,
)

from .optimizer import (
    prepare_player_pool,
    optimise_squad_with_xi,
    optimise_squad_sequential_with_captain,
    choose_starting_xi,
    choose_captain_and_vice,
    best_single_transfer,
    validate_squad,
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
        sp.add_argument(
            "--score-type",
            choices=[
                "next",
                "horizon",
                "validated",
            ],
            default="horizon",
            help=(
                "Projection to use: next, horizon, "
                "or the validated minutes-and-shrinkage "
                "live model."
            ),
        )
        sp.add_argument(
            "--recent-gws",
            type=int,
            default=5,
            help=(
                "Completed gameweeks used for expected "
                "minutes (default: 5)."
            ),
        )

        sp.add_argument(
            "--shrinkage-matches",
            type=float,
            default=5.0,
            help=(
                "Position-prior strength in full-match "
                "equivalents (default: 5)."
            ),
        )
    p_opt = sub.add_parser("optimise", help="Optimise a fresh 15-man squad")
    common_opts(p_opt)

    p_opt.add_argument(
        "--bench-weight",
        type=float,
        default=0.10,
        help=(
            "Relative value assigned to substitute "
            "points (default: 0.10)"
        ),
    )

    p_opt.add_argument(
        "--selection-strategy",
        choices=[
            "sequential",
            "joint",
        ],
        default="sequential",
        help=(
            "Squad selection method; joint is "
            "experimental (default: sequential)"
        ),
    )
    p_opt.add_argument(
        "--track",
        action="store_true",
        help=(
            "Append this recommendation "
            "to the weekly tracking files"
        ),
    )

    p_opt.add_argument(
        "--tracking-dir",
        default="tracking",
        help=(
            "Directory containing persistent "
            "weekly tracking CSVs"
        ),
    )

    p_xi = sub.add_parser("xi", help="Choose best starting XI for a given 15-man squad (CSV input)")
    common_opts(p_xi)
    p_xi.add_argument("--squad-file", required=True, help="Path to a CSV with 'name' column (optional 'team' column)")


    p_tr = sub.add_parser(
        "transfers",
        help=(
            "Suggest best single transfer "
            "for current squad (CSV input)"
        ),
    )

    common_opts(p_tr)

    p_tr.add_argument(
        "--squad-file",
        required=True,
        help=(
            "Path to a CSV with 'name' column "
            "(optional 'team' column)"
        ),
    )

    p_tr.add_argument(
        "--bank",
        type=float,
        default=0.0,
        help="Money in the bank (millions)",
    )

    p_tr.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top K transfer suggestions",
    )

    # Command used after a gameweek finishes.
    p_track = sub.add_parser(
        "track",
        help=(
            "Score tracked recommendations "
            "for finished gameweeks"
        ),
    )

    p_track.add_argument(
        "--tracking-dir",
        default="tracking",
        help=(
            "Directory containing persistent "
            "weekly tracking CSVs"
        ),
    )

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
    Read a squad CSV containing:
      - required: name
      - optional: team or team_name
    """

    df = pd.read_csv(path)
    cols_lower = {
        column.lower(): column
        for column in df.columns
    }

    if "name" not in cols_lower:
        raise ValueError(
            "Input squad CSV must contain a 'name' column."
        )

    name_column = cols_lower["name"]

    output = pd.DataFrame({
        "name": df[name_column].astype(str).str.strip()
    })

    team_key = next(
        (
            key
            for key in ("team", "team_name")
            if key in cols_lower
        ),
        None,
    )

    if team_key:
        team_column = cols_lower[team_key]

        output["team"] = (
            df[team_column]
            .astype(str)
            .str.strip()
        )

        missing_team = (
            output["team"].eq("")
            | output["team"].str.lower().eq("nan")
        )

        output.loc[missing_team, "team"] = None

    return output


def run_optimise(args):
    ensure_outdir(args.out_dir)

    pool_df = prepare_player_pool(
        allow_flagged=args.allow_flagged,
        min_play_chance=args.min_play_chance,
        horizon=args.horizon,
        include_validated_live=(
            args.score_type == "validated"
        ),
        recent_gws=args.recent_gws,
        shrinkage_matches=(
            args.shrinkage_matches
        ),
    )


    plan_arguments = dict(
        players=pool_df,
        budget_m=args.budget,
        score_type=args.score_type,
        max_per_team=MAX_PER_TEAM,
        required_counts=REQUIRED_COUNTS,
        lock_names=set(args.lock),
        ban_names=set(args.ban),
        min_from_team=None,
        verbose=False,
    )

    if args.selection_strategy == "joint":
        plan = optimise_squad_with_xi(
            **plan_arguments,
            bench_weight=args.bench_weight,
        )
    else:
        plan = optimise_squad_sequential_with_captain(
            **plan_arguments,
        )

    selected = plan.squad
    xi = plan.starting_xi.copy()
    bench = plan.bench.copy()
    score_col = plan.score_column

    # Assign each selected player a role.
    xi_ids = set(plan.starting_xi["id"])

    show = selected.copy()

    show["role"] = "BENCH"

    show.loc[
        show["id"].isin(xi_ids),
        "role",
    ] = "XI"

    show.loc[
        show["id"] == plan.vice_captain["id"],
        "role",
    ] = "VICE"

    show.loc[
        show["id"] == plan.captain["id"],
        "role",
    ] = "CAPTAIN"

    show = show.sort_values(
        [
            "pos_code",
            "team_name",
            "name",
        ]
    ).copy()

    # Save the full 15-player plan.
    squad_columns = [
        "name",
        "team_name",
        "pos_code",
        "role",
        "cost_m",
        "status",
        "avail_mult",
        "exp_points_next",
        "exp_points_h",
        "exp_points_next_avail",
        "exp_points_h_avail",
    ]

    # Add the live minutes-and-shrinkage diagnostics
    # when the validated model is selected.
    if args.score_type == "validated":
        squad_columns.extend(
            [
                "recent_minutes",
                "recent_fixture_count",
                "expected_minutes_share",
                "position_points_per_90",
                "shrunk_points_per_90",
                "next_fixture_weight",
                "exp_points_validated_live",
                "validated_score_source",
            ]
        )

    squad_out = os.path.join(
        args.out_dir,
        f"optimised_squad_{args.score_type}.csv",
    )
    show[squad_columns].to_csv(
        squad_out,
        index=False,
    )

    # Save the starting XI with captaincy flags.
    xi["is_captain"] = (
        xi["id"] == plan.captain["id"]
    )

    xi["is_vice_captain"] = (
        xi["id"] == plan.vice_captain["id"]
    )

    xi = xi.sort_values(
        [
            "pos_code",
            "team_name",
            "name",
        ]
    )

    bench = bench.sort_values(
        [
            "pos_code",
            "team_name",
            "name",
        ]
    )

    xi_out = os.path.join(
        args.out_dir,
        f"starting_xi_{args.score_type}.csv",
    )

    bench_out = os.path.join(
        args.out_dir,
        f"bench_{args.score_type}.csv",
    )

    xi.to_csv(
        xi_out,
        index=False,
    )

    bench.to_csv(
        bench_out,
        index=False,
    )

    # Display the complete squad.
    display_squad = show.copy()

    display_squad["cost_m"] = (
        display_squad["cost_m"]
        .map(lambda value: f"£{value:.1f}m")
    )

    display_squad[score_col] = (
        display_squad[score_col]
        .map(lambda value: f"{value:.2f}")
    )

    print(
        f"\n[info] Strategy: {plan.strategy}"
        f" | Scoring: {args.score_type}"
        f" ({score_col})"
    )

    print("\nOptimised squad:")

    print_table(
        display_squad[
            [
                "name",
                "team_name",
                "pos_code",
                "role",
                "cost_m",
                score_col,
            ]
        ]
    )

    # Display the starting XI.
    display_xi = xi.copy()

    display_xi[score_col] = (
        display_xi[score_col]
        .map(lambda value: f"{value:.2f}")
    )

    print("\nStarting XI:")

    print_table(
        display_xi[
            [
                "name",
                "team_name",
                "pos_code",
                "is_captain",
                "is_vice_captain",
                score_col,
            ]
        ]
    )

    print("\nPlan summary:")

    print(
        f"  Cost: £{plan.total_cost:.1f}m "
        f"/ £{args.budget:.1f}m"
    )

    print(
        f"  Captain: {plan.captain['name']} "
        f"({float(plan.captain[score_col]):.2f})"
    )

    print(
        f"  Vice-captain: "
        f"{plan.vice_captain['name']} "
        f"({float(plan.vice_captain[score_col]):.2f})"
    )

    print(
        f"  XI expected points: "
        f"{plan.predicted_xi_points:.2f}"
    )

    print(
        f"  Captain bonus: "
        f"{plan.predicted_captain_bonus:.2f}"
    )

    print(
        f"  Projected FPL points: "
        f"{plan.predicted_fpl_points:.2f}"
    )

    if args.selection_strategy == "joint":
        print(
            f"  Joint objective value: "
            f"{plan.objective_value:.2f}"
        )

        print(
            f"  Bench weight: "
            f"{args.bench_weight:.2f}"
        )

    print(f"\n[saved] {squad_out}")
    print(f"[saved] {xi_out}")
    print(f"[saved] {bench_out}")

    if args.track:
        run_id = record_recommendation(
            plan=plan,
            score_type=args.score_type,
            tracking_dir=args.tracking_dir,
            recent_gws=args.recent_gws,
            shrinkage_matches=(
                args.shrinkage_matches
            ),
            bench_weight=(
                args.bench_weight
                if args.selection_strategy == "joint"
                else 0.0
            ),
        )

        print(
            f"[tracked] {run_id}"
        )

        tracking_summary = os.path.join(
            args.tracking_dir,
            "weekly_summary.csv",
        )

        print(
            f"[tracking] {tracking_summary}"
        )



def run_xi(args):
    ensure_outdir(args.out_dir)
    names_df = read_squad_csv(args.squad_file)
    pool_df = prepare_player_pool(
        allow_flagged=args.allow_flagged,
        min_play_chance=args.min_play_chance,
        horizon=args.horizon,
        include_validated_live=(
            args.score_type == "validated"
        ),
        recent_gws=args.recent_gws,
        shrinkage_matches=(
            args.shrinkage_matches
        ),
    )              # <-- pass DataFrame
    current_df = map_names_to_pool(
        pool_df,
        names_df,
    )

    validate_squad(current_df)
    xi, bench, xi_pts = choose_starting_xi(current_df, args.score_type)

    score_col = {
        "next": "exp_points_next_avail",
        "horizon": "exp_points_h_avail",
        "validated": (
            "exp_points_validated_live"
        ),
    }[args.score_type]
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
    names_df = read_squad_csv(args.squad_file)
    pool_df = prepare_player_pool(
        allow_flagged=args.allow_flagged,
        min_play_chance=args.min_play_chance,
        horizon=args.horizon,
        include_validated_live=(
            args.score_type == "validated"
        ),
        recent_gws=args.recent_gws,
        shrinkage_matches=(
            args.shrinkage_matches
        ),
    )              # <-- pass DataFrame

    current_df = map_names_to_pool(
        pool_df,
        names_df,
    )

    validate_squad(current_df)

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

def run_track(args):
    settled = settle_finished_recommendations(
        args.tracking_dir
    )

    if settled.empty:
        print(
            "No pending recommendations belong "
            "to a finished gameweek."
        )
        return

    print(
        "\nScored weekly recommendations:"
    )

    print_table(
        settled[
            [
                "run_id",
                "actual_xi_points",
                "actual_bench_points",
                "actual_captain_bonus",
                "actual_fpl_points",
                "prediction_error",
                "captain_counted",
                "autosub_count",
            ]
        ]
    )

    print(
        "[updated] "
        f"{os.path.join(
            args.tracking_dir,
            'weekly_summary.csv',
        )}"
    )

def main():
    args = parse_args()

    try:
        if args.cmd == "optimise":
            run_optimise(args)

        elif args.cmd == "xi":
            run_xi(args)

        elif args.cmd == "transfers":
            run_transfers(args)

        elif args.cmd == "track":
            run_track(args)

        else:
            raise SystemExit(2)

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise SystemExit(
            f"[error] {exc}"
        ) from exc

if __name__ == "__main__":
    main()
