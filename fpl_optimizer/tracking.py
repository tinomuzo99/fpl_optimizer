from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .fpl import (
    fetch_bootstrap,
    fetch_event_live,
    get_season_label,
)
from .optimizer import SquadPlan


PICKS_FILE = "weekly_picks.csv"
SUMMARY_FILE = "weekly_summary.csv"


def record_recommendation(
    plan: SquadPlan,
    score_type: str,
    tracking_dir: str | Path,
    recent_gws: int = 5,
    shrinkage_matches: float = 5.0,
    bench_weight: float = 0.10,
) -> str:
    """Append one recommendation to the weekly tracking CSV files."""

    required_metadata = {
        "target_gw",
        "season",
    }

    missing_metadata = (
        required_metadata
        - set(plan.squad.columns)
    )

    if missing_metadata:
        raise ValueError(
            "Player pool is missing tracking metadata: "
            + ", ".join(sorted(missing_metadata))
        )

    target_gw = int(
        plan.squad["target_gw"].iloc[0]
    )

    season = str(
        plan.squad["season"].iloc[0]
    )

    created_at = (
        datetime.now(timezone.utc).isoformat()
    )

    run_id = (
        f"{season}-GW{target_gw:02d}-"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-"
        f"{uuid4().hex[:8]}"
    )

    xi_ids = set(
        plan.starting_xi["id"].astype(int)
    )

    # Determine bench order.
    # The substitute goalkeeper gets order 0.
    # Outfield substitutes are ordered by predicted points.
    bench = plan.bench.copy()

    bench["bench_order"] = pd.NA

    outfield_bench = bench.loc[
        bench["pos_code"] != "GK"
    ].sort_values(
        plan.score_column,
        ascending=False,
    )

    for order, index in enumerate(
        outfield_bench.index,
        start=1,
    ):
        bench.loc[
            index,
            "bench_order",
        ] = order

    bench.loc[
        bench["pos_code"] == "GK",
        "bench_order",
    ] = 0

    bench_order = (
        bench.set_index("id")["bench_order"]
        .to_dict()
    )

    pick_rows = []

    for _, player in plan.squad.iterrows():
        player_id = int(player["id"])

        if player_id == int(plan.captain["id"]):
            role = "CAPTAIN"

        elif player_id == int(
            plan.vice_captain["id"]
        ):
            role = "VICE"

        elif player_id in xi_ids:
            role = "XI"

        else:
            role = "BENCH"

        pick_rows.append(
            {
                "run_id": run_id,
                "created_at_utc": created_at,
                "season": season,
                "target_gw": target_gw,
                "strategy": plan.strategy,
                "score_type": score_type,
                "player_id": player_id,
                "name": player["name"],
                "team_name": player["team_name"],
                "pos_code": player["pos_code"],
                "role": role,
                "bench_order": bench_order.get(
                    player_id,
                    pd.NA,
                ),
                "cost_m": float(
                    player["cost_m"]
                ),
                "predicted_points": float(
                    player[plan.score_column]
                ),
                "actual_minutes": pd.NA,
                "actual_points": pd.NA,
                "counted_in_final_xi": pd.NA,
            }
        )

    summary_row = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "season": season,
        "target_gw": target_gw,
        "strategy": plan.strategy,
        "score_type": score_type,
        "recent_gws": recent_gws,
        "shrinkage_matches": shrinkage_matches,
        "bench_weight": bench_weight,
        "status": "pending",
        "predicted_xi_points": (
            plan.predicted_xi_points
        ),
        "predicted_captain_bonus": (
            plan.predicted_captain_bonus
        ),
        "predicted_fpl_points": (
            plan.predicted_fpl_points
        ),
        "actual_xi_points": pd.NA,
        "actual_bench_points": pd.NA,
        "actual_captain_bonus": pd.NA,
        "actual_fpl_points": pd.NA,
        "prediction_error": pd.NA,
        "captain_counted": pd.NA,
        "autosub_count": pd.NA,
        "scored_at_utc": pd.NA,
    }

    tracking_path = Path(tracking_dir)

    tracking_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    _append_csv(
        tracking_path / PICKS_FILE,
        pd.DataFrame(pick_rows),
    )

    _append_csv(
        tracking_path / SUMMARY_FILE,
        pd.DataFrame([summary_row]),
    )

    return run_id


def settle_finished_recommendations(
    tracking_dir: str | Path,
) -> pd.DataFrame:
    """
    Score every pending recommendation whose gameweek
    has been marked finished by FPL.
    """

    tracking_path = Path(tracking_dir)

    picks_path = (
        tracking_path / PICKS_FILE
    )

    summary_path = (
        tracking_path / SUMMARY_FILE
    )

    if (
        not picks_path.exists()
        or not summary_path.exists()
    ):
        raise FileNotFoundError(
            "Tracking files were not found in "
            f"{tracking_path}. Run optimise "
            "with --track first."
        )

    picks = pd.read_csv(picks_path)
    summary = pd.read_csv(summary_path)

    picks["counted_in_final_xi"] = (
        picks["counted_in_final_xi"]
        .astype("object")
    )

    summary["captain_counted"] = (
        summary["captain_counted"]
        .astype("object")
    )

    summary["scored_at_utc"] = (
        summary["scored_at_utc"]
        .astype("object")
    )

    bootstrap = fetch_bootstrap()

    current_season = get_season_label(
        bootstrap
    )

    finished_gws = {
        int(event["id"])
        for event in bootstrap["events"]
        if bool(
            event.get("finished", False)
        )
    }

    # Only score pending recommendations belonging
    # to the currently active season.
    pending = summary.loc[
        (summary["status"] == "pending")
        & (
            summary["season"].astype(str)
            == current_season
        )
    ].copy()

    eligible_gws = sorted(
        set(
            pending["target_gw"].astype(int)
        )
        & finished_gws
    )

    live_by_gw = {
        gameweek: _actual_player_stats(
            fetch_event_live(gameweek)
        )
        for gameweek in eligible_gws
    }

    settled_rows = []

    for summary_index, run in pending.iterrows():
        gameweek = int(
            run["target_gw"]
        )

        if gameweek not in live_by_gw:
            continue

        run_mask = (
            picks["run_id"]
            == run["run_id"]
        )

        run_picks = (
            picks.loc[run_mask]
            .copy()
        )

        scored_picks, result = (
            score_tracked_team(
                run_picks,
                live_by_gw[gameweek],
            )
        )

        for column in (
            "actual_minutes",
            "actual_points",
            "counted_in_final_xi",
        ):
            picks.loc[
                run_mask,
                column,
            ] = scored_picks[
                column
            ].to_numpy()

        scored_at = (
            datetime.now(timezone.utc)
            .isoformat()
        )

        prediction_error = (
            result["actual_fpl_points"]
            - float(
                run["predicted_fpl_points"]
            )
        )

        updates = {
            "status": "scored",
            "actual_xi_points": (
                result["actual_xi_points"]
            ),
            "actual_bench_points": (
                result["actual_bench_points"]
            ),
            "actual_captain_bonus": (
                result[
                    "actual_captain_bonus"
                ]
            ),
            "actual_fpl_points": (
                result["actual_fpl_points"]
            ),
            "prediction_error": (
                prediction_error
            ),
            "captain_counted": (
                result["captain_counted"]
            ),
            "autosub_count": (
                result["autosub_count"]
            ),
            "scored_at_utc": scored_at,
        }

        for column, value in updates.items():
            summary.loc[
                summary_index,
                column,
            ] = value

        settled_rows.append(
            {
                "run_id": run["run_id"],
                **updates,
            }
        )

    picks.to_csv(
        picks_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    return pd.DataFrame(settled_rows)


def score_tracked_team(
    picks: pd.DataFrame,
    actual_stats: dict[
        int,
        dict[str, float],
    ],
) -> tuple[pd.DataFrame, dict]:
    """
    Calculate the recommended team's actual FPL score,
    including legal auto-substitutions and captain fallback.
    """

    if len(picks) != 15:
        raise ValueError(
            "A tracked run must contain 15 players; "
            f"found {len(picks)}."
        )

    scored = (
        picks.copy()
        .reset_index(drop=True)
    )

    scored["actual_minutes"] = (
        scored["player_id"].map(
            lambda player_id: (
                actual_stats.get(
                    int(player_id),
                    {},
                ).get(
                    "minutes",
                    0.0,
                )
            )
        )
    )

    scored["actual_points"] = (
        scored["player_id"].map(
            lambda player_id: (
                actual_stats.get(
                    int(player_id),
                    {},
                ).get(
                    "points",
                    0.0,
                )
            )
        )
    )

    starting_mask = scored["role"].isin(
        [
            "XI",
            "CAPTAIN",
            "VICE",
        ]
    )

    current_xi = list(
        scored.index[starting_mask]
    )

    autosub_count = 0

    # Goalkeeper substitution.
    starting_gk = next(
        index
        for index in current_xi
        if scored.loc[
            index,
            "pos_code",
        ] == "GK"
    )

    bench_gks = scored.index[
        (scored["role"] == "BENCH")
        & (
            scored["pos_code"]
            == "GK"
        )
    ].tolist()

    if (
        scored.loc[
            starting_gk,
            "actual_minutes",
        ] == 0
        and bench_gks
    ):
        bench_gk = bench_gks[0]

        if scored.loc[
            bench_gk,
            "actual_minutes",
        ] > 0:
            current_xi[
                current_xi.index(
                    starting_gk
                )
            ] = bench_gk

            autosub_count += 1

    # Process outfield substitutes according
    # to their saved bench order.
    outfield_bench = scored.loc[
        (scored["role"] == "BENCH")
        & (
            scored["pos_code"]
            != "GK"
        )
    ].sort_values("bench_order")

    for bench_index in outfield_bench.index:
        if scored.loc[
            bench_index,
            "actual_minutes",
        ] <= 0:
            continue

        replaceable = [
            starter_index
            for starter_index in current_xi
            if (
                scored.loc[
                    starter_index,
                    "pos_code",
                ] != "GK"
                and scored.loc[
                    starter_index,
                    "actual_minutes",
                ] == 0
            )
        ]

        replacement_index = next(
            (
                starter_index
                for starter_index
                in replaceable
                if _legal_formation_after_swap(
                    scored,
                    current_xi,
                    starter_index,
                    bench_index,
                )
            ),
            None,
        )

        if replacement_index is not None:
            current_xi[
                current_xi.index(
                    replacement_index
                )
            ] = bench_index

            autosub_count += 1

    scored["counted_in_final_xi"] = False

    scored.loc[
        current_xi,
        "counted_in_final_xi",
    ] = True

    actual_xi_points = float(
        scored.loc[
            current_xi,
            "actual_points",
        ].sum()
    )

    actual_bench_points = float(
        scored.loc[
            ~scored.index.isin(
                current_xi
            ),
            "actual_points",
        ].sum()
    )

    captain_index = scored.index[
        scored["role"] == "CAPTAIN"
    ][0]

    vice_index = scored.index[
        scored["role"] == "VICE"
    ][0]

    if scored.loc[
        captain_index,
        "actual_minutes",
    ] > 0:
        captain_counted = str(
            scored.loc[
                captain_index,
                "name",
            ]
        )

        captain_bonus = float(
            scored.loc[
                captain_index,
                "actual_points",
            ]
        )

    elif scored.loc[
        vice_index,
        "actual_minutes",
    ] > 0:
        captain_counted = str(
            scored.loc[
                vice_index,
                "name",
            ]
        )

        captain_bonus = float(
            scored.loc[
                vice_index,
                "actual_points",
            ]
        )

    else:
        captain_counted = "none"
        captain_bonus = 0.0

    return scored, {
        "actual_xi_points": (
            actual_xi_points
        ),
        "actual_bench_points": (
            actual_bench_points
        ),
        "actual_captain_bonus": (
            captain_bonus
        ),
        "actual_fpl_points": (
            actual_xi_points
            + captain_bonus
        ),
        "captain_counted": (
            captain_counted
        ),
        "autosub_count": (
            autosub_count
        ),
    }


def _actual_player_stats(
    payload: dict,
) -> dict[int, dict[str, float]]:
    """Extract minutes and points from an event-live response."""

    return {
        int(element["id"]): {
            "minutes": float(
                element.get(
                    "stats",
                    {},
                ).get(
                    "minutes",
                    0,
                )
                or 0
            ),
            "points": float(
                element.get(
                    "stats",
                    {},
                ).get(
                    "total_points",
                    0,
                )
                or 0
            ),
        }
        for element
        in payload.get(
            "elements",
            [],
        )
    }


def _legal_formation_after_swap(
    picks: pd.DataFrame,
    current_xi: list[int],
    starter_index: int,
    bench_index: int,
) -> bool:
    """Check whether an outfield auto-sub leaves a legal formation."""

    candidate_xi = [
        bench_index
        if index == starter_index
        else index
        for index in current_xi
    ]

    counts = (
        picks.loc[
            candidate_xi,
            "pos_code",
        ]
        .value_counts()
        .to_dict()
    )

    return (
        counts.get("GK", 0) == 1
        and counts.get("DEF", 0) >= 3
        and counts.get("MID", 0) >= 2
        and counts.get("FWD", 0) >= 1
    )


def _append_csv(
    path: Path,
    rows: pd.DataFrame,
) -> None:
    """Append rows to a CSV, creating it if necessary."""

    if path.exists():
        existing = pd.read_csv(path)

        rows = pd.concat(
            [
                existing,
                rows,
            ],
            ignore_index=True,
        )

    rows.to_csv(
        path,
        index=False,
    )