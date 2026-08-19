from __future__ import annotations
from typing import Dict, Optional, Set, Literal
import pandas as pd
import pulp
from dataclasses import dataclass
from .config import REQUIRED_COUNTS, MAX_PER_TEAM, DIFF_TO_MULT, AVAIL_SCALE
from .fpl import (
    fetch_bootstrap,
    fetch_event_live,
    fetch_fixtures,
    build_player_table,
    add_horizon_expected_points,
    compute_team_gw_weights,
    get_next_gw_and_window,
    get_season_label,
)

@dataclass(frozen=True)
class SquadPlan:
    strategy: str
    squad: pd.DataFrame
    starting_xi: pd.DataFrame
    bench: pd.DataFrame
    captain: pd.Series
    vice_captain: pd.Series
    total_cost: float
    predicted_xi_points: float
    predicted_captain_bonus: float
    predicted_fpl_points: float
    objective_value: float
    score_column: str


# ---------------------------------------------------------------------------
# Build the player pool with BOTH metrics:
#   - exp_points_next: next GW (from ep_next/ppg fallback)
#   - exp_points_h:    weighted sum over the horizon window
# Also produce availability-adjusted versions for optimisation:
#   - exp_points_next_avail
#   - exp_points_h_avail
# ---------------------------------------------------------------------------

def prepare_player_pool(
    allow_flagged: bool,
    min_play_chance: int,
    horizon: int,
    include_validated_live: bool = False,
    recent_gws: int = 5,
    shrinkage_matches: float = 5.0,
) -> pd.DataFrame:
    bootstrap = fetch_bootstrap()
    fixtures_df = fetch_fixtures()

    pool = build_player_table(
        bootstrap,
        allow_flagged=allow_flagged,
        min_play_chance=min_play_chance,
    )

    pool = add_horizon_expected_points(
        pool,
        bootstrap,
        fixtures_df,
        horizon=horizon,
        diff_to_mult=DIFF_TO_MULT,
    )

    elements = pd.DataFrame(
        bootstrap["elements"]
    )

    pool = pool.merge(
        elements[
            [
                "id",
                "status",
            ]
        ],
        on="id",
        how="left",
    )

    pool["avail_mult"] = (
        pool["status"]
        .map(AVAIL_SCALE)
        .fillna(0.8)
    )

    pool["exp_points_next_avail"] = (
        pool["exp_points_next"]
        * pool["avail_mult"]
    )

    pool["exp_points_h_avail"] = (
        pool["exp_points_h"]
        * pool["avail_mult"]
    )

    if include_validated_live:
        pool = add_validated_live_score(
            pool,
            bootstrap,
            fixtures_df,
            recent_gws=recent_gws,
            shrinkage_matches=shrinkage_matches,
        )

    # Tracking metadata must be added for every score type.
    next_gw, _ = get_next_gw_and_window(
        bootstrap,
        horizon=1,
    )

    pool["season"] = get_season_label(
        bootstrap
    )

    pool["target_gw"] = next_gw

    return pool




def _score_col(
    score_type: Literal[
        "next",
        "horizon",
        "validated",
    ],
) -> str:
    if score_type == "next":
        return "exp_points_next_avail"

    if score_type == "horizon":
        return "exp_points_h_avail"

    if score_type == "validated":
        return "exp_points_validated_live"

    raise ValueError(
        f"Unknown score type: {score_type}"
    )

def add_validated_live_score(
    players: pd.DataFrame,
    bootstrap: dict,
    fixtures_df: pd.DataFrame,
    recent_gws: int = 5,
    shrinkage_matches: float = 5.0,
    minimum_completed_gws: int = 3,
) -> pd.DataFrame:
    """
    Add the cross-season validated one-gameweek projection.

    Formula:
        shrunk points per 90
        × expected minutes share
        × next fixture weight
        × availability
    """

    if recent_gws <= 0:
        raise ValueError(
            "recent_gws must be greater than zero."
        )

    if shrinkage_matches < 0:
        raise ValueError(
            "shrinkage_matches must not be negative."
        )

    df = players.copy()

    events = pd.DataFrame(
        bootstrap["events"]
    )

    completed_gws = sorted(
        int(gameweek)
        for gameweek in events.loc[
            events["finished"].fillna(False),
            "id",
        ].tolist()
    )

    # Our historical backtests started at GW4. Until three
    # gameweeks are complete, use the official projection.
    if len(completed_gws) < minimum_completed_gws:
        df["recent_minutes"] = 0.0
        df["recent_fixture_count"] = 0
        df["expected_minutes_share"] = 0.0
        df["position_points_per_90"] = 0.0
        df["shrunk_points_per_90"] = 0.0
        df["next_fixture_weight"] = 0.0

        df["exp_points_validated_live"] = (
            df["exp_points_next_avail"]
        )

        df["validated_score_source"] = (
            "official_fallback_early_season"
        )

        return df

    recent_window = completed_gws[
        -recent_gws:
    ]

    recent_frames = []

    for gameweek in recent_window:
        payload = fetch_event_live(
            gameweek
        )

        rows = []

        for element in payload.get(
            "elements",
            [],
        ):
            stats = element.get(
                "stats",
                {},
            )

            rows.append(
                {
                    "id": int(
                        element["id"]
                    ),
                    "minutes": float(
                        stats.get(
                            "minutes",
                            0,
                        )
                        or 0
                    ),
                }
            )

        if rows:
            recent_frames.append(
                pd.DataFrame(rows)
            )

    if not recent_frames:
        raise RuntimeError(
            "Completed gameweeks were found, but "
            "the FPL event-live API returned no "
            "recent player statistics."
        )

    recent_stats = pd.concat(
        recent_frames,
        ignore_index=True,
    )

    recent_minutes = (
        recent_stats
        .groupby(
            "id",
            as_index=False,
        )["minutes"]
        .sum()
        .rename(
            columns={
                "minutes": "recent_minutes"
            }
        )
    )

    # Cumulative season performance is available in
    # bootstrap-static.
    elements = pd.DataFrame(
        bootstrap["elements"]
    )

    season_stats = elements[
        [
            "id",
            "total_points",
            "minutes",
        ]
    ].copy()

    season_stats.rename(
        columns={
            "total_points": "season_points",
            "minutes": "season_minutes",
        },
        inplace=True,
    )

    for column in (
        "season_points",
        "season_minutes",
    ):
        season_stats[column] = (
            pd.to_numeric(
                season_stats[column],
                errors="coerce",
            )
            .fillna(0.0)
        )

    df = df.merge(
        season_stats,
        on="id",
        how="left",
    )

    df = df.merge(
        recent_minutes,
        on="id",
        how="left",
    )

    history_columns = [
        "season_points",
        "season_minutes",
        "recent_minutes",
    ]

    df[history_columns] = (
        df[history_columns]
        .fillna(0.0)
    )

    # Count actual fixtures so blank and double gameweeks
    # are handled correctly.
    recent_fixtures = fixtures_df.loc[
        fixtures_df["event"].isin(
            recent_window
        )
    ].copy()

    team_fixture_rows = []

    for _, fixture in (
        recent_fixtures.iterrows()
    ):
        team_fixture_rows.append(
            int(fixture["team_h"])
        )

        team_fixture_rows.append(
            int(fixture["team_a"])
        )

    fixture_counts = (
        pd.Series(
            team_fixture_rows,
            dtype="int64",
        )
        .value_counts()
        .to_dict()
    )

    df["recent_fixture_count"] = (
        df["team_id"]
        .map(fixture_counts)
        .fillna(0)
        .astype(int)
    )

    recent_minute_capacity = (
        90.0
        * df["recent_fixture_count"]
    )

    df["expected_minutes_share"] = (
        df["recent_minutes"]
        / recent_minute_capacity.where(
            recent_minute_capacity > 0
        )
    ).fillna(0.0).clip(
        lower=0.0,
        upper=1.0,
    )

    # Calculate pooled points per 90 for each position.
    position_totals = (
        df
        .groupby(
            "pos_code",
            as_index=False,
        )
        .agg(
            position_points=(
                "season_points",
                "sum",
            ),
            position_minutes=(
                "season_minutes",
                "sum",
            ),
        )
    )

    position_minute_denominator = (
        position_totals[
            "position_minutes"
        ]
        .where(
            position_totals[
                "position_minutes"
            ] > 0
        )
    )

    position_totals[
        "position_points_per_90"
    ] = (
        90.0
        * position_totals[
            "position_points"
        ]
        / position_minute_denominator
    ).fillna(0.0)

    position_rates = (
        position_totals
        .set_index(
            "pos_code"
        )["position_points_per_90"]
        .to_dict()
    )

    df["position_points_per_90"] = (
        df["pos_code"]
        .map(position_rates)
        .fillna(0.0)
    )

    # Shrink each player towards their positional average.
    full_match_equivalents = (
        df["season_minutes"]
        / 90.0
    )

    shrinkage_denominator = (
        full_match_equivalents
        + shrinkage_matches
    )

    df["shrunk_points_per_90"] = (
        (
            df["season_points"]
            + (
                shrinkage_matches
                * df[
                    "position_points_per_90"
                ]
            )
        )
        / shrinkage_denominator.where(
            shrinkage_denominator > 0
        )
    ).fillna(0.0)

    # Calculate the upcoming fixture weight.
    next_gw, _ = get_next_gw_and_window(
        bootstrap,
        horizon=1,
    )

    next_weights = (
        compute_team_gw_weights(
            bootstrap,
            fixtures_df,
            [next_gw],
            DIFF_TO_MULT,
        )
    )

    df["next_fixture_weight"] = (
        df["team_id"]
        .map(
            lambda team_id: float(
                next_weights.get(
                    (
                        int(team_id),
                        next_gw,
                    ),
                    0.0,
                )
            )
        )
    )

    model_score = (
        df["shrunk_points_per_90"]
        * df["expected_minutes_share"]
        * df["next_fixture_weight"]
        * df["avail_mult"]
    )

    has_player_history = (
        (df["season_minutes"] > 0)
        & (df["recent_fixture_count"] > 0)
    )

    # New signings and players without current-season
    # history retain the official FPL projection.
    df["exp_points_validated_live"] = (
        model_score.where(
            has_player_history,
            df["exp_points_next_avail"],
        )
        .clip(lower=0.0)
    )

    df["validated_score_source"] = (
        "minutes_shrunk_fixture"
    )

    df.loc[
        ~has_player_history,
        "validated_score_source",
    ] = "official_fallback_no_player_history"

    return df


def validate_squad(
    squad_df: pd.DataFrame,
    required_counts: Dict[str, int] = REQUIRED_COUNTS,
    max_per_team: int = MAX_PER_TEAM,
) -> None:
    """Raise ValueError when a squad does not satisfy official FPL squad rules."""

    errors = []
    expected_size = sum(required_counts.values())

    if len(squad_df) != expected_size:
        errors.append(
            f"expected {expected_size} players, but found {len(squad_df)}"
        )

    required_columns = {"name", "team_name", "pos_code"}
    missing_columns = sorted(required_columns - set(squad_df.columns))

    if missing_columns:
        errors.append(
            "missing required columns: " + ", ".join(missing_columns)
        )
    else:
        if squad_df[
            ["name", "team_name", "pos_code"]
        ].isna().any().any():
            errors.append(
                "player name, team and position values must not be blank"
            )

        actual_counts = squad_df["pos_code"].value_counts().to_dict()

        for position, expected in required_counts.items():
            actual = int(actual_counts.get(position, 0))

            if actual != expected:
                errors.append(
                    f"expected {expected} {position}, but found {actual}"
                )

        unexpected_positions = sorted(
            set(actual_counts) - set(required_counts)
        )

        if unexpected_positions:
            errors.append(
                "unexpected positions: " + ", ".join(unexpected_positions)
            )

        team_counts = squad_df["team_name"].value_counts()
        over_limit = team_counts[team_counts > max_per_team]

        if not over_limit.empty:
            details = ", ".join(
                f"{team} ({count})"
                for team, count in over_limit.items()
            )
            errors.append(
                f"more than {max_per_team} players from a club: {details}"
            )

    identity_columns = (
        ["id"]
        if "id" in squad_df.columns
        else ["name", "team_name"]
    )

    if all(column in squad_df.columns for column in identity_columns):
        duplicate_mask = squad_df.duplicated(
            identity_columns,
            keep=False,
        )

        if duplicate_mask.any():
            duplicates = sorted(
                squad_df.loc[
                    duplicate_mask, "name"
                ].astype(str).unique()
            )
            errors.append(
                "duplicate players: " + ", ".join(duplicates)
            )

    if errors:
        raise ValueError(
            "Invalid FPL squad: " + "; ".join(errors) + "."
        )
# ---------------------------------------------------------------------------
# Optimise best 15 given budget/constraints and a chosen score metric
# ---------------------------------------------------------------------------
def optimise_squad(
    players: pd.DataFrame,
    budget_m: float,
    score_type: Literal["next", "horizon", "validated"] = "horizon",
    max_per_team: int = MAX_PER_TEAM,
    required_counts: Dict[str, int] = REQUIRED_COUNTS,
    lock_names: Optional[Set[str]] = None,
    ban_names: Optional[Set[str]] = None,
    min_from_team: Optional[Dict[str, int]] = None,
    verbose: bool = False,
):
    score = _score_col(score_type)
    df = players.copy()

    # Apply bans
    bans = {
        name.strip().lower()
        for name in (ban_names or set())
    }

    if bans:
        banned_mask = df["name"].str.lower().apply(
            lambda player_name: any(
                banned_name in player_name
                for banned_name in bans
            )
        )
        df = df[~banned_mask].copy()

    model = pulp.LpProblem(
        "FPL_Squad_Optimisation",
        sense=pulp.LpMaximize,
    )

    indices = list(df.index)

    selected_variables = {
        index: pulp.LpVariable(
            f"x_{index}",
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )
        for index in indices
    }

    # Objective
    model += pulp.lpSum(
        df.loc[index, score] * selected_variables[index]
        for index in indices
    )

    # Budget
    model += (
        pulp.lpSum(
            df.loc[index, "cost_m"] * selected_variables[index]
            for index in indices
        )
        <= budget_m
    )

    # Exactly 15 players
    model += (
        pulp.lpSum(
            selected_variables[index]
            for index in indices
        )
        == sum(required_counts.values())
    )

    # Exact positional requirements
    for position, required in required_counts.items():
        position_indices = df.index[
            df["pos_code"] == position
        ]

        model += (
            pulp.lpSum(
                selected_variables[index]
                for index in position_indices
            )
            == required
        )

    # Maximum players per club
    for team_name in sorted(df["team_name"].unique()):
        team_indices = df.index[
            df["team_name"] == team_name
        ]

        model += (
            pulp.lpSum(
                selected_variables[index]
                for index in team_indices
            )
            <= max_per_team
        )

    # Locked players
    locks = {
        name.strip().lower()
        for name in (lock_names or set())
    }

    for locked_name in locks:
        matches = df.index[
            df["name"]
            .str.lower()
            .str.contains(locked_name, regex=False)
        ]

        if len(matches) == 0:
            raise ValueError(
                f"Locked player '{locked_name}' did not "
                "match an eligible player."
            )

        model += (
            pulp.lpSum(
                selected_variables[index]
                for index in matches
            )
            >= 1
        )

    # Minimum players from particular clubs
    if min_from_team:
        for team_name, minimum in min_from_team.items():
            team_indices = df.index[
                df["team_name"].str.lower()
                == team_name.lower()
            ]

            if len(team_indices) == 0:
                raise ValueError(
                    f"Minimum-team constraint '{team_name}' "
                    "did not match an eligible club."
                )

            model += (
                pulp.lpSum(
                    selected_variables[index]
                    for index in team_indices
                )
                >= int(minimum)
            )

    status = model.solve(
        pulp.PULP_CBC_CMD(msg=verbose)
    )

    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(
            "No valid squad satisfies the budget, position, "
            "club, lock and ban constraints. "
            "No FPL rules were relaxed."
        )

    selection_mask = [
        bool(selected_variables[index].value())
        for index in df.index
    ]

    selected = (
        df.loc[selection_mask]
        .copy()
        .reset_index(drop=True)
    )

    # Defensive validation of solver output
    validate_squad(
        selected,
        required_counts=required_counts,
        max_per_team=max_per_team,
    )

    total_cost = float(selected["cost_m"].sum())
    total_points = float(selected[score].sum())

    return (
        "strict",
        selected,
        total_cost,
        total_points,
        score,
    )

    def _build_and_solve(df, budget, max_team, req_counts, locks, bans, min_team, loosen_positions=False):
        # Apply bans
        bans = {b.strip().lower() for b in (bans or set())}
        if bans:
            mask = df["name"].str.lower().apply(lambda nm: any(b in nm for b in bans))
            df = df[~mask].copy()

        model = pulp.LpProblem("FPL_Squad_Optimisation", sense=pulp.LpMaximize)
        idxs = list(df.index)
        x = {i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in idxs}

        # Objective: maximise chosen metric
        model += pulp.lpSum(df.loc[i, score] * x[i] for i in idxs)

        # Constraints
        model += pulp.lpSum(df.loc[i, "cost_m"] * x[i] for i in idxs) <= budget, "Budget"
        model += pulp.lpSum(x[i] for i in idxs) == 15, "SquadSize"

        for pos, req in req_counts.items():
            pos_idx = df.index[df["pos_code"] == pos]
            if loosen_positions:
                model += pulp.lpSum(x[i] for i in pos_idx) >= req, f"MinCount_{pos}"
            else:
                model += pulp.lpSum(x[i] for i in pos_idx) == req, f"Count_{pos}"

        for team in sorted(df["team_name"].unique()):
            t_idx = df.index[df["team_name"] == team]
            model += pulp.lpSum(x[i] for i in t_idx) <= max_team, f"Max3_{team}"

        # Locks
        locks = {l.lower() for l in (locks or set())}
        for ln in locks:
            matches = df.index[df["name"].str.lower().str.contains(ln)]
            if len(matches) > 0:
                model += pulp.lpSum(x[i] for i in matches) >= 1, f"Lock_{ln}"

        # Minimum from team
        if min_team:
            for team, k in min_team.items():
                t_idx = df.index[df["team_name"].str.lower() == team.lower()]
                if len(t_idx) > 0:
                    model += pulp.lpSum(x[i] for i in t_idx) >= int(k), f"Min_{team}"

        status = model.solve(pulp.PULP_CBC_CMD(msg=verbose))
        return pulp.LpStatus[status], df, x, model

    tries = [
        ("strict", players.copy(), budget_m, max_per_team, required_counts, lock_names or set(), ban_names or set(), min_from_team or {}, False),
        ("team_cap_4", players.copy(), budget_m, 4,               required_counts, set(), set(), {}, False),
    ]
    relaxed_req = {k: v for k, v in required_counts.items()}
    tries.append(("pos_loosen", players.copy(), budget_m, max_per_team, relaxed_req, set(), set(), {}, True))

    for tag, df, bud, mteam, reqs, locks, bans, mint, loosen in tries:
        status, df_used, x, model = _build_and_solve(df, bud, mteam, reqs, locks, bans, mint, loosen_positions=loosen)
        if status == "Optimal":
            sel_mask = [bool(x[i].value()) for i in df_used.index]
            sel = df_used.loc[sel_mask].copy()
            sel_cost   = float(sel["cost_m"].sum())
            sel_points = float(sel[score].sum())
            return tag, sel.reset_index(drop=True), sel_cost, sel_points, score

    raise RuntimeError("Infeasible after relaxations — check constraints.")

def optimise_squad_with_xi(
    players: pd.DataFrame,
    budget_m: float,
    score_type: Literal["next", "horizon", "validated"] = "horizon",
    bench_weight: float = 0.10,
    max_per_team: int = MAX_PER_TEAM,
    required_counts: Dict[str, int] = REQUIRED_COUNTS,
    lock_names: Optional[Set[str]] = None,
    ban_names: Optional[Set[str]] = None,
    min_from_team: Optional[Dict[str, int]] = None,
    verbose: bool = False,
) -> SquadPlan:
    """Jointly choose a legal squad, starting XI and captain."""

    if budget_m <= 0:
        raise ValueError("Budget must be greater than zero.")

    if not 0 <= bench_weight <= 1:
        raise ValueError(
            "bench_weight must be between 0 and 1."
        )

    score = _score_col(score_type)
    df = players.copy().reset_index(drop=True)

    required_columns = {
        "name",
        "team_name",
        "pos_code",
        "cost_m",
        score,
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Player pool is missing required columns: "
            + ", ".join(missing_columns)
            + "."
        )

    df[score] = pd.to_numeric(
        df[score],
        errors="coerce",
    ).fillna(0.0)

    df["cost_m"] = pd.to_numeric(
        df["cost_m"],
        errors="coerce",
    )

    df = df.loc[
        df["cost_m"].notna()
        & (df["cost_m"] > 0)
    ].copy()

    bans = {
        name.strip().lower()
        for name in (ban_names or set())
        if name.strip()
    }

    if bans:
        banned = (
            df["name"]
            .astype(str)
            .str.lower()
            .apply(
                lambda player_name: any(
                    name in player_name
                    for name in bans
                )
            )
        )

        df = df.loc[~banned].copy()

    df = df.reset_index(drop=True)
    indices = list(df.index)

    model = pulp.LpProblem(
        "FPL_Joint_Squad_XI",
        sense=pulp.LpMaximize,
    )

    squad_var = {
        i: pulp.LpVariable(
            f"squad_{i}",
            0,
            1,
            cat=pulp.LpBinary,
        )
        for i in indices
    }

    xi_var = {
        i: pulp.LpVariable(
            f"xi_{i}",
            0,
            1,
            cat=pulp.LpBinary,
        )
        for i in indices
    }

    captain_var = {
        i: pulp.LpVariable(
            f"captain_{i}",
            0,
            1,
            cat=pulp.LpBinary,
        )
        for i in indices
    }

    # Starting players score normally, the captain receives
    # one additional copy of their score, and substitutes
    # receive a small bench weight.
    model += pulp.lpSum(
        float(df.loc[i, score])
        * (
            xi_var[i]
            + captain_var[i]
            + bench_weight
            * (squad_var[i] - xi_var[i])
        )
        for i in indices
    )

    model += (
        pulp.lpSum(
            float(df.loc[i, "cost_m"])
            * squad_var[i]
            for i in indices
        )
        <= budget_m,
        "Budget",
    )

    model += (
        pulp.lpSum(
            squad_var[i]
            for i in indices
        )
        == sum(required_counts.values()),
        "SquadSize",
    )

    model += (
        pulp.lpSum(
            xi_var[i]
            for i in indices
        )
        == 11,
        "StartingXI",
    )

    model += (
        pulp.lpSum(
            captain_var[i]
            for i in indices
        )
        == 1,
        "Captain",
    )

    for i in indices:
        model += (
            xi_var[i] <= squad_var[i],
            f"XIInSquad_{i}",
        )

        model += (
            captain_var[i] <= xi_var[i],
            f"CaptainInXI_{i}",
        )

    # Exact squad-position requirements.
    for position, required in required_counts.items():
        position_indices = df.index[
            df["pos_code"] == position
        ]

        model += (
            pulp.lpSum(
                squad_var[i]
                for i in position_indices
            )
            == required,
            f"Squad_{position}",
        )

    def xi_position(position: str):
        position_indices = df.index[
            df["pos_code"] == position
        ]

        return pulp.lpSum(
            xi_var[i]
            for i in position_indices
        )

    # Legal starting formation.
    model += xi_position("GK") == 1, "XI_GK"
    model += xi_position("DEF") >= 3, "XI_Min_DEF"
    model += xi_position("MID") >= 2, "XI_Min_MID"
    model += xi_position("FWD") >= 1, "XI_Min_FWD"

    # Maximum three players per club.
    for team_name in sorted(
        df["team_name"].dropna().unique()
    ):
        team_indices = df.index[
            df["team_name"] == team_name
        ]

        model += (
            pulp.lpSum(
                squad_var[i]
                for i in team_indices
            )
            <= max_per_team,
            f"Club_{team_name}",
        )

    locks = {
        name.strip().lower()
        for name in (lock_names or set())
        if name.strip()
    }

    for locked_name in locks:
        matches = df.index[
            df["name"]
            .astype(str)
            .str.lower()
            .str.contains(
                locked_name,
                regex=False,
            )
        ]

        if len(matches) == 0:
            raise ValueError(
                f"Locked player '{locked_name}' did not "
                "match an eligible player."
            )

        model += (
            pulp.lpSum(
                squad_var[i]
                for i in matches
            )
            >= 1,
            f"Lock_{locked_name}",
        )

    if min_from_team:
        for team_name, minimum in min_from_team.items():
            team_indices = df.index[
                df["team_name"]
                .astype(str)
                .str.lower()
                == team_name.lower()
            ]

            if len(team_indices) == 0:
                raise ValueError(
                    f"Minimum-team constraint '{team_name}' "
                    "did not match an eligible club."
                )

            model += (
                pulp.lpSum(
                    squad_var[i]
                    for i in team_indices
                )
                >= int(minimum),
                f"Minimum_{team_name}",
            )

    status = model.solve(
        pulp.PULP_CBC_CMD(
            msg=verbose,
        )
    )

    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(
            "No valid joint squad and starting XI satisfies "
            "the budget, position, formation, club, lock and "
            "ban constraints. No FPL rules were relaxed."
        )

    squad_indices = [
        i
        for i in indices
        if squad_var[i].value() > 0.5
    ]

    xi_indices = [
        i
        for i in indices
        if xi_var[i].value() > 0.5
    ]

    captain_index = next(
        i
        for i in indices
        if captain_var[i].value() > 0.5
    )

    xi_index_set = set(xi_indices)

    squad = (
        df.loc[squad_indices]
        .copy()
        .reset_index(drop=True)
    )

    starting_xi = (
        df.loc[xi_indices]
        .copy()
        .reset_index(drop=True)
    )

    bench = (
        df.loc[
            [
                i
                for i in squad_indices
                if i not in xi_index_set
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    captain = df.loc[
        captain_index
    ].copy()

    if (
        "id" in starting_xi.columns
        and "id" in captain.index
    ):
        vice_candidates = starting_xi.loc[
            starting_xi["id"]
            != captain["id"]
        ]
    else:
        vice_candidates = starting_xi.loc[
            starting_xi["name"]
            != captain["name"]
        ]

    vice_captain = (
        vice_candidates
        .sort_values(
            score,
            ascending=False,
        )
        .iloc[0]
        .copy()
    )

    validate_squad(
        squad,
        required_counts=required_counts,
        max_per_team=max_per_team,
    )

    predicted_xi_points = float(
        starting_xi[score].sum()
    )

    predicted_captain_bonus = float(
        captain[score]
    )

    predicted_fpl_points = (
        predicted_xi_points
        + predicted_captain_bonus
    )

    raw_objective_value = pulp.value(
        model.objective
    )

    objective_value = (
        float(raw_objective_value)
        if raw_objective_value is not None
        else 0.0
    )

    return SquadPlan(
        strategy="joint_xi_captain",
        squad=squad,
        starting_xi=starting_xi,
        bench=bench,
        captain=captain,
        vice_captain=vice_captain,
        total_cost=float(
            squad["cost_m"].sum()
        ),
        predicted_xi_points=predicted_xi_points,
        predicted_captain_bonus=(
            predicted_captain_bonus
        ),
        predicted_fpl_points=(
            predicted_fpl_points
        ),
        objective_value=objective_value,
        score_column=score,
    )

# ---------------------------------------------------------------------------
# Starting XI using chosen metric
# ---------------------------------------------------------------------------
def choose_starting_xi(
    squad_df: pd.DataFrame,
    score_type: Literal["next", "horizon", "validated"] = "horizon",
):
    score = _score_col(score_type)
    df = squad_df.copy()

    validate_squad(df)

    model = pulp.LpProblem("FPL_Starting_XI", sense=pulp.LpMaximize)
    idxs = list(df.index)
    y = {i: pulp.LpVariable(f"y_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in idxs}

    model += pulp.lpSum(df.loc[i, score] * y[i] for i in idxs)
    model += pulp.lpSum(y[i] for i in idxs) == 11, "XI_size"

    def idx_pos(p): return df.index[df["pos_code"] == p]
    model += pulp.lpSum(y[i] for i in idx_pos("GK")) == 1
    model += pulp.lpSum(y[i] for i in idx_pos("DEF")) >= 3
    model += pulp.lpSum(y[i] for i in idx_pos("MID")) >= 2
    model += pulp.lpSum(y[i] for i in idx_pos("FWD")) >= 1

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"XI Solver status: {pulp.LpStatus[status]}")

    start = df[[bool(v.value()) for v in y.values()]].copy()
    bench = df[[not bool(v.value()) for v in y.values()]].copy()
    total_points = float(start[score].sum())
    return start.reset_index(drop=True), bench.reset_index(drop=True), total_points

def choose_captain_and_vice(
    starting_xi: pd.DataFrame,
    score_type: Literal[
        "next",
        "horizon",
    ] = "horizon",
) -> tuple[pd.Series, pd.Series]:
    """Choose the two highest-projected starters."""

    if len(starting_xi) < 2:
        raise ValueError(
            "At least two starters are required "
            "for captaincy."
        )

    score = _score_col(score_type)

    if score not in starting_xi.columns:
        raise ValueError(
            f"Starting XI is missing score column "
            f"'{score}'."
        )

    ordered = starting_xi.sort_values(
        score,
        ascending=False,
    )

    captain = ordered.iloc[0].copy()
    vice_captain = ordered.iloc[1].copy()

    return captain, vice_captain

def optimise_squad_sequential_with_captain(
    players: pd.DataFrame,
    budget_m: float,
    score_type: Literal[
        "next",
        "horizon",
    ] = "horizon",
    max_per_team: int = MAX_PER_TEAM,
    required_counts: Dict[
        str,
        int,
    ] = REQUIRED_COUNTS,
    lock_names: Optional[Set[str]] = None,
    ban_names: Optional[Set[str]] = None,
    min_from_team: Optional[
        Dict[str, int]
    ] = None,
    verbose: bool = False,
) -> SquadPlan:
    """Use the established squad-then-XI method."""

    (
        _,
        squad,
        total_cost,
        squad_points,
        score,
    ) = optimise_squad(
        players=players,
        budget_m=budget_m,
        score_type=score_type,
        max_per_team=max_per_team,
        required_counts=required_counts,
        lock_names=lock_names,
        ban_names=ban_names,
        min_from_team=min_from_team,
        verbose=verbose,
    )

    (
        starting_xi,
        bench,
        predicted_xi_points,
    ) = choose_starting_xi(
        squad,
        score_type=score_type,
    )

    captain, vice_captain = (
        choose_captain_and_vice(
            starting_xi,
            score_type,
        )
    )

    captain_bonus = float(
        captain[score]
    )

    return SquadPlan(
        strategy="sequential_xi_captain",
        squad=squad,
        starting_xi=starting_xi,
        bench=bench,
        captain=captain,
        vice_captain=vice_captain,
        total_cost=total_cost,
        predicted_xi_points=(
            predicted_xi_points
        ),
        predicted_captain_bonus=(
            captain_bonus
        ),
        predicted_fpl_points=(
            predicted_xi_points
            + captain_bonus
        ),
        objective_value=squad_points,
        score_column=score,
    )



# ---------------------------------------------------------------------------
# Transfer recommendations respecting chosen metric
# ---------------------------------------------------------------------------
def max_three_ok(after_df: pd.DataFrame) -> bool:
    return (after_df["team_name"].value_counts() <= 3).all()

def affordable_after_swap(current_df: pd.DataFrame, out_row: pd.Series, in_row: pd.Series, bank_m: float) -> bool:
    current_cost = float(current_df["cost_m"].sum())
    new_cost = current_cost - float(out_row["cost_m"]) + float(in_row["cost_m"])
    return new_cost <= current_cost + bank_m + 1e-9

def apply_swap(current_df: pd.DataFrame, out_idx: int, in_row: pd.Series) -> pd.DataFrame:
    new_df = current_df.copy()
    new_df.loc[out_idx] = in_row
    return new_df

def shortlist_in_pool(pool: pd.DataFrame, pos: str, exclude_names: set, max_price: float | None, score: str) -> pd.DataFrame:
    df = pool.copy()
    df = df[df["pos_code"] == pos]
    if exclude_names:
        ex = set(n.lower() for n in exclude_names)
        df = df[~df["name"].str.lower().isin(ex)]
    if max_price is not None:
        df = df[df["cost_m"] <= max_price + 1e-9]
    return df.sort_values(score, ascending=False)

def best_single_transfer(
    pool_df: pd.DataFrame,
    current_df: pd.DataFrame,
    bank_m: float = 0.0,
    top_k: int = 10,
    score_type: Literal["next", "horizon", "validated"] = "horizon",
) -> pd.DataFrame:
    """
    Evaluate best single-transfer upgrades using the provided score_type.

    pool_df / current_df must contain:
      - 'name', 'team_name', 'pos_code', 'cost_m'
      - 'exp_points_next_avail', 'exp_points_h_avail'
    """
    score = _score_col(score_type)
    validate_squad(current_df)

    base_xi_pts = choose_starting_xi(
        current_df,
        score_type,
    )[2]

    already = set(current_df["name"].str.lower().tolist())
    results = []

    for out_idx, out_row in current_df.reset_index().iterrows():
        pos = out_row["pos_code"]
        candidates = shortlist_in_pool(pool_df, pos, exclude_names=already,
                                       max_price=out_row["cost_m"] + bank_m, score=score)
        for _, in_row in candidates.iterrows():
            if not affordable_after_swap(current_df, out_row, in_row, bank_m):
                continue
            after = apply_swap(current_df, out_idx, in_row)
            if not max_three_ok(after):
                continue
            new_xi_pts = choose_starting_xi(after, score_type)[2]
            delta = new_xi_pts - base_xi_pts
            if delta > 0.01:
                results.append({
                    "score_type": score_type,
                    "out": out_row["name"], "out_team": out_row["team_name"], "out_pos": pos, "out_cost": float(out_row["cost_m"]),
                    "in": in_row["name"],  "in_team":  in_row["team_name"],  "in_pos":  in_row["pos_code"], "in_cost":  float(in_row["cost_m"]),
                    "delta_pts": float(delta)
                })

    if not results:
        return pd.DataFrame(columns=["score_type","out","out_team","out_pos","out_cost","in","in_team","in_pos","in_cost","delta_pts"])

    recs = pd.DataFrame(results).sort_values("delta_pts", ascending=False).head(top_k)
    return recs.reset_index(drop=True)
