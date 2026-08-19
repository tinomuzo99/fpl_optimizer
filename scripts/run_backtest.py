from __future__ import annotations

import argparse

from fpl_optimizer.backtest import (
    load_historical_season,
    run_historical_backtest,
    save_backtest_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a leakage-aware historical FPL backtest."
    )

    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--data-dir", default="data/historical")
    parser.add_argument(
        "--out-dir",
        default="backtests/results/2024-25",
    )
    parser.add_argument("--start-gw", type=int, default=4)
    parser.add_argument("--end-gw", type=int, default=38)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--recent-gws", type=int, default=5)
    parser.add_argument(
        "--shrinkage-matches",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    if args.start_gw < 2:
        parser.error("--start-gw must be at least 2")

    if args.end_gw < args.start_gw:
        parser.error(
            "--end-gw must be greater than or equal to --start-gw"
        )

    if args.recent_gws <= 0:
        parser.error("--recent-gws must be greater than zero")

    if args.shrinkage_matches < 0:
        parser.error("--shrinkage-matches must not be negative")

    gameweeks, fixtures, teams = load_historical_season(
        args.data_dir,
        args.season,
    )

    result = run_historical_backtest(
        gameweeks,
        fixtures,
        teams,
        range(args.start_gw, args.end_gw + 1),
        budget_m=args.budget,
        top_k=args.top_k,
        recent_gws=args.recent_gws,
        shrinkage_matches=args.shrinkage_matches,
    )

    save_backtest_results(
        result,
        args.out_dir,
    )

    print("\nBacktest summary")
    print(result.summary.to_string(index=False))

    print("\nPaired XI comparisons (model_a minus model_b)")
    print(result.paired_comparisons.to_string(index=False))

    print(f"\nSaved results to {args.out_dir}")


if __name__ == "__main__":
    main()