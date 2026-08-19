from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .config import DIFF_TO_MULT
from .optimizer import choose_starting_xi, optimise_squad


ModelName = Literal[
    "prior_ppg",
    "fixture_ppg",
    "minutes_shrunk_fixture",
]

VALID_POSITIONS = {"GK", "DEF", "MID", "FWD"}

REQUIRED_GW_COLUMNS = {
    "element",
    "name",
    "position",
    "team",
    "total_points",
    "minutes",
    "value",
    "GW",
}

REQUIRED_FIXTURE_COLUMNS = {
    "event",
    "team_h",
    "team_a",
    "team_h_difficulty",
    "team_a_difficulty",
}


@dataclass(frozen=True)
class BacktestResult:
    player_predictions: pd.DataFrame
    gameweek_results: pd.DataFrame
    summary: pd.DataFrame
    paired_comparisons: pd.DataFrame


def load_historical_season(
    data_root: str | Path,
    season: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load one downloaded historical FPL season."""

    season_dir = Path(data_root) / season

    gw_path = season_dir / "merged_gw.csv"
    fixtures_path = season_dir / "fixtures.csv"
    teams_path = season_dir / "teams.csv"

    missing = [
        str(path)
        for path in (
            gw_path,
            fixtures_path,
            teams_path,
        )
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing historical data file(s): "
            + ", ".join(missing)
        )

    gameweeks = pd.read_csv(
        gw_path,
        low_memory=False,
    )

    fixtures = pd.read_csv(
        fixtures_path,
        low_memory=False,
    )

    teams = pd.read_csv(
        teams_path,
        low_memory=False,
    )

    _require_columns(
        gameweeks,
        REQUIRED_GW_COLUMNS,
        "merged_gw.csv",
    )

    _require_columns(
        fixtures,
        REQUIRED_FIXTURE_COLUMNS,
        "fixtures.csv",
    )

    _require_columns(
        teams,
        {"id", "name"},
        "teams.csv",
    )

    numeric_columns = (
        "element",
        "total_points",
        "minutes",
        "value",
        "GW",
    )

    for column in numeric_columns:
        gameweeks[column] = pd.to_numeric(
            gameweeks[column],
            errors="coerce",
        )

    gameweeks = gameweeks.dropna(
        subset=["element", "GW"]
    ).copy()

    gameweeks["element"] = (
        gameweeks["element"].astype(int)
    )

    gameweeks["GW"] = gameweeks["GW"].astype(int)

    fixtures["event"] = pd.to_numeric(
        fixtures["event"],
        errors="coerce",
    )

    return gameweeks, fixtures, teams


def build_gameweek_pool(
    gameweeks: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    target_gw: int,
    model: ModelName = "fixture_ppg",
    recent_gws: int = 5,
    shrinkage_matches: float = 5.0,
) -> pd.DataFrame:
    """
    Reconstruct a player pool for one historical gameweek.

    Player performance from the target gameweek and later is
    excluded from all predictions. The target-gameweek data is
    only used afterwards to evaluate the predictions.
    """

    if target_gw < 2:
        raise ValueError(
            "target_gw must be at least 2; "
            "GW1 has no current-season history."
        )

    valid_models = {
        "prior_ppg",
        "fixture_ppg",
        "minutes_shrunk_fixture",
    }

    if model not in valid_models:
        raise ValueError(
            f"Unknown backtest model: {model}"
        )

    if recent_gws <= 0:
        raise ValueError(
            "recent_gws must be greater than zero."
        )

    if shrinkage_matches < 0:
        raise ValueError(
            "shrinkage_matches must not be negative."
        )

    observed = gameweeks.loc[
        gameweeks["GW"] <= target_gw
    ].copy()

    if observed.empty:
        raise ValueError(
            f"No historical rows are available "
            f"through GW{target_gw}."
        )

    # Recover the most recent player metadata available by
    # the target gameweek. During a double gameweek, the
    # player's metadata is repeated, so the final row is enough.
    sort_columns = ["GW"]

    if "kickoff_time" in observed.columns:
        sort_columns.append("kickoff_time")

    latest = (
        observed
        .sort_values(sort_columns)
        .groupby(
            "element",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    team_id_to_name = (
        teams
        .set_index("id")["name"]
        .to_dict()
    )

    target_fixtures = fixtures.loc[
        fixtures["event"] == target_gw
    ].copy()

    teams_playing_ids = (
        set(target_fixtures["team_h"])
        | set(target_fixtures["team_a"])
    )

    teams_playing = {
        team_id_to_name.get(int(team_id))
        for team_id in teams_playing_ids
        if pd.notna(team_id)
    }

    # If a player is absent from a club that plays in the
    # target gameweek, they are normally no longer part of
    # that historical player pool.
    #
    # Players from clubs with a blank gameweek are retained
    # using their latest previous metadata.
    active_mask = (
        (latest["GW"] == target_gw)
        | (~latest["team"].isin(teams_playing))
    )

    latest = latest.loc[
        active_mask
    ].copy()

    # Only gameweeks strictly before the target gameweek are
    # used to construct predictive features.
    prior = gameweeks.loc[
        gameweeks["GW"] < target_gw
    ].copy()

    prior["played"] = (
        prior["minutes"].fillna(0) > 0
    ).astype(int)

    history = (
        prior
        .groupby(
            "element",
            as_index=False,
        )
        .agg(
            prior_points=(
                "total_points",
                "sum",
            ),
            prior_appearances=(
                "played",
                "sum",
            ),
            prior_minutes=(
                "minutes",
                "sum",
            ),
        )
    )

    history["prior_ppg"] = np.where(
        history["prior_appearances"] > 0,
        (
            history["prior_points"]
            / history["prior_appearances"]
        ),
        0.0,
    )

    # Calculate the average points-per-90 rate for each
    # position, using only information before the target GW.
    position_rates = _position_points_per_90(
        prior
    )

    # Recent minutes are used to estimate the probability
    # and extent of participation.
    recent = prior.loc[
        prior["GW"] >= target_gw - recent_gws
    ].copy()

    recent_minutes = (
        recent
        .groupby(
            "element",
            as_index=False,
        )
        .agg(
            recent_minutes=(
                "minutes",
                "sum",
            ),
            recent_fixture_rows=(
                "GW",
                "size",
            ),
        )
    )

    # Target-gameweek points are outcomes, not prediction
    # inputs. Multiple rows are summed for double gameweeks.
    actual = (
        gameweeks.loc[
            gameweeks["GW"] == target_gw
        ]
        .groupby(
            "element",
            as_index=False,
        )["total_points"]
        .sum()
        .rename(
            columns={
                "total_points": "actual_points"
            }
        )
    )

    fixture_weights = _team_fixture_weights(
        target_fixtures,
        team_id_to_name,
    )

    pool = latest.merge(
        history,
        on="element",
        how="left",
    )

    pool = pool.merge(
        recent_minutes,
        on="element",
        how="left",
    )

    pool = pool.merge(
        actual,
        on="element",
        how="left",
    )

    pool["prior_points"] = (
        pool["prior_points"].fillna(0.0)
    )

    pool["prior_appearances"] = (
        pool["prior_appearances"]
        .fillna(0)
        .astype(int)
    )

    pool["prior_minutes"] = (
        pool["prior_minutes"].fillna(0.0)
    )

    pool["prior_ppg"] = (
        pool["prior_ppg"].fillna(0.0)
    )

    pool["recent_minutes"] = (
        pool["recent_minutes"].fillna(0.0)
    )

    pool["recent_fixture_rows"] = (
        pool["recent_fixture_rows"]
        .fillna(0)
        .astype(int)
    )

    pool["actual_points"] = (
        pool["actual_points"].fillna(0.0)
    )

    pool["fixture_weight"] = (
        pool["team"]
        .map(fixture_weights)
        .fillna(0.0)
    )

    pool["position_points_per_90"] = (
        pool["position"]
        .astype(str)
        .str.upper()
        .map(position_rates)
        .fillna(0.0)
    )

    # Express accumulated minutes as full-match equivalents.
    full_match_equivalents = (
        pool["prior_minutes"] / 90.0
    )

    shrinkage_denominator = (
        full_match_equivalents
        + shrinkage_matches
    )

    # Pull a player's points-per-90 rate towards the average
    # for their position. Players with little history receive
    # stronger shrinkage than established players.
    pool["shrunk_points_per_90"] = np.where(
        shrinkage_denominator > 0,
        (
            pool["prior_points"]
            + (
                shrinkage_matches
                * pool["position_points_per_90"]
            )
        )
        / shrinkage_denominator,
        0.0,
    )

    # Estimate expected minutes as the proportion of the
    # available 90 minutes played across recent fixture rows.
    pool["expected_minutes_share"] = np.where(
        pool["recent_fixture_rows"] > 0,
        (
            pool["recent_minutes"]
            / (
                90.0
                * pool["recent_fixture_rows"]
            )
        ),
        0.0,
    )

    pool["expected_minutes_share"] = (
        pool["expected_minutes_share"]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    if model == "prior_ppg":
        predicted = pool["prior_ppg"]

    elif model == "fixture_ppg":
        predicted = (
            pool["prior_ppg"]
            * pool["fixture_weight"]
        )

    else:
        predicted = (
            pool["shrunk_points_per_90"]
            * pool["expected_minutes_share"]
            * pool["fixture_weight"]
        )

    pool["id"] = (
        pool["element"].astype(int)
    )

    pool["team_name"] = (
        pool["team"].astype(str)
    )

    pool["pos_code"] = (
        pool["position"]
        .astype(str)
        .str.upper()
    )

    pool["cost_m"] = (
        pd.to_numeric(
            pool["value"],
            errors="coerce",
        )
        / 10.0
    )

    # These names match the columns expected by the existing
    # squad and starting-XI optimisation functions.
    pool["exp_points_next"] = (
        predicted
        .clip(lower=0.0)
        .fillna(0.0)
    )

    pool["exp_points_next_avail"] = (
        pool["exp_points_next"]
    )

    pool["exp_points_h"] = (
        pool["exp_points_next"]
    )

    pool["exp_points_h_avail"] = (
        pool["exp_points_next"]
    )

    pool = pool.loc[
        pool["pos_code"].isin(
            VALID_POSITIONS
        )
        & pool["cost_m"].notna()
        & (pool["cost_m"] > 0)
    ].copy()

    keep_columns = [
        "id",
        "name",
        "team_name",
        "pos_code",
        "cost_m",
        "prior_points",
        "prior_appearances",
        "prior_minutes",
        "prior_ppg",
        "recent_minutes",
        "recent_fixture_rows",
        "position_points_per_90",
        "shrunk_points_per_90",
        "expected_minutes_share",
        "fixture_weight",
        "exp_points_next",
        "exp_points_next_avail",
        "exp_points_h",
        "exp_points_h_avail",
        "actual_points",
    ]

    return (
        pool[keep_columns]
        .drop_duplicates("id")
        .reset_index(drop=True)
    )


def run_historical_backtest(
    gameweeks: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    gameweek_range: Iterable[int],
    models: Iterable[ModelName] = (
        "prior_ppg",
        "fixture_ppg",
        "minutes_shrunk_fixture",
    ),
    budget_m: float = 100.0,
    top_k: int = 20,
    recent_gws: int = 5,
    shrinkage_matches: float = 5.0,
) -> BacktestResult:
    """
    Run fresh-squad, one-gameweek-ahead backtests.

    A new legal squad is constructed for every model and
    gameweek. This does not yet simulate transfers between
    gameweeks.
    """

    if budget_m <= 0:
        raise ValueError(
            "budget_m must be greater than zero."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if recent_gws <= 0:
        raise ValueError(
            "recent_gws must be greater than zero."
        )

    if shrinkage_matches < 0:
        raise ValueError(
            "shrinkage_matches must not be negative."
        )

    prediction_frames: list[pd.DataFrame] = []
    gameweek_rows: list[dict] = []

    for target_gw in gameweek_range:
        for model in models:
            pool = build_gameweek_pool(
                gameweeks=gameweeks,
                fixtures=fixtures,
                teams=teams,
                target_gw=int(target_gw),
                model=model,
                recent_gws=recent_gws,
                shrinkage_matches=shrinkage_matches,
            )

            (
                _,
                squad,
                total_cost,
                predicted_squad_points,
                _,
            ) = optimise_squad(
                pool,
                budget_m=budget_m,
                score_type="next",
            )

            (
                starting_xi,
                bench,
                predicted_xi_points,
            ) = choose_starting_xi(
                squad,
                score_type="next",
            )

            selected_ids = set(
                squad["id"]
            )

            xi_ids = set(
                starting_xi["id"]
            )

            player_output = pool.copy()

            player_output.insert(
                0,
                "model",
                model,
            )

            player_output.insert(
                0,
                "GW",
                int(target_gw),
            )

            player_output["selected_squad"] = (
                player_output["id"].isin(
                    selected_ids
                )
            )

            player_output["selected_xi"] = (
                player_output["id"].isin(
                    xi_ids
                )
            )

            prediction_frames.append(
                player_output
            )

            metrics = calculate_prediction_metrics(
                pool,
                top_k=top_k,
            )

            gameweek_rows.append(
                {
                    "GW": int(target_gw),
                    "model": model,
                    **metrics,
                    "squad_cost": float(
                        total_cost
                    ),
                    "predicted_squad_points": float(
                        predicted_squad_points
                    ),
                    "predicted_xi_points": float(
                        predicted_xi_points
                    ),
                    "actual_squad_points": float(
                        squad[
                            "actual_points"
                        ].sum()
                    ),
                    "actual_xi_points": float(
                        starting_xi[
                            "actual_points"
                        ].sum()
                    ),
                    "actual_bench_points": float(
                        bench[
                            "actual_points"
                        ].sum()
                    ),
                }
            )

    if not gameweek_rows:
        raise ValueError(
            "The requested gameweek range is empty."
        )

    player_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    gameweek_results = (
        pd.DataFrame(
            gameweek_rows
        )
        .sort_values(
            ["model", "GW"]
        )
        .reset_index(drop=True)
    )

    summary = summarise_backtest(
        gameweek_results
    )

    paired_comparisons = (
        paired_model_comparisons(
            gameweek_results
        )
    )

    return BacktestResult(
        player_predictions=player_predictions,
        gameweek_results=gameweek_results,
        summary=summary,
        paired_comparisons=paired_comparisons,
    )


def calculate_prediction_metrics(
    player_pool: pd.DataFrame,
    top_k: int = 20,
) -> dict[str, float]:
    """Calculate player-level prediction metrics."""

    predicted = (
        player_pool[
            "exp_points_next_avail"
        ].astype(float)
    )

    actual = (
        player_pool[
            "actual_points"
        ].astype(float)
    )

    errors = predicted - actual

    k = min(
        top_k,
        len(player_pool),
    )

    predicted_top = set(
        player_pool
        .nlargest(
            k,
            "exp_points_next_avail",
        )["id"]
    )

    actual_top = set(
        player_pool
        .nlargest(
            k,
            "actual_points",
        )["id"]
    )

    if (
        predicted.nunique() > 1
        and actual.nunique() > 1
    ):
        rank_correlation = (
            predicted
            .rank(method="average")
            .corr(
                actual.rank(
                    method="average"
                )
            )
        )
    else:
        rank_correlation = np.nan

    return {
        "mae": float(
            errors.abs().mean()
        ),
        "rmse": float(
            np.sqrt(
                (errors**2).mean()
            )
        ),
        "spearman": (
            float(rank_correlation)
            if pd.notna(rank_correlation)
            else np.nan
        ),
        "top_k_recall": float(
            len(
                predicted_top
                & actual_top
            )
            / k
        ),
    }


def summarise_backtest(
    gameweek_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate results across gameweeks for each model."""

    metric_columns = [
        "mae",
        "rmse",
        "spearman",
        "top_k_recall",
        "actual_xi_points",
        "actual_squad_points",
        "actual_bench_points",
        "predicted_xi_points",
    ]

    summary = (
        gameweek_results
        .groupby(
            "model",
            as_index=False,
        )
        .agg(
            gameweeks=(
                "GW",
                "nunique",
            ),
            **{
                f"mean_{column}": (
                    column,
                    "mean",
                )
                for column
                in metric_columns
            },
        )
    )

    return (
        summary
        .sort_values(
            "mean_actual_xi_points",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def paired_model_comparisons(
    gameweek_results: pd.DataFrame,
    bootstrap_samples: int = 10_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compare model XI points on identical gameweeks.

    The confidence interval is calculated by bootstrapping
    paired gameweek differences.
    """

    if bootstrap_samples <= 0:
        raise ValueError(
            "bootstrap_samples must be greater than zero."
        )

    pivot = gameweek_results.pivot(
        index="GW",
        columns="model",
        values="actual_xi_points",
    )

    preferred_pairs = [
        (
            "fixture_ppg",
            "prior_ppg",
        ),
        (
            "minutes_shrunk_fixture",
            "prior_ppg",
        ),
        (
            "minutes_shrunk_fixture",
            "fixture_ppg",
        ),
    ]

    available_models = list(
        pivot.columns
    )

    pairs = [
        pair
        for pair in preferred_pairs
        if (
            pair[0] in available_models
            and pair[1] in available_models
        )
    ]

    already_added = {
        frozenset(pair)
        for pair in pairs
    }

    for model_a, model_b in combinations(
        available_models,
        2,
    ):
        pair_key = frozenset(
            (model_a, model_b)
        )

        if pair_key not in already_added:
            pairs.append(
                (model_a, model_b)
            )

    random_generator = (
        np.random.default_rng(
            random_seed
        )
    )

    rows = []

    for model_a, model_b in pairs:
        paired = (
            pivot[
                [model_a, model_b]
            ]
            .dropna()
        )

        differences = (
            paired[model_a]
            - paired[model_b]
        ).to_numpy(dtype=float)

        if len(differences) == 0:
            continue

        sampled_means = (
            random_generator.choice(
                differences,
                size=(
                    bootstrap_samples,
                    len(differences),
                ),
                replace=True,
            )
            .mean(axis=1)
        )

        lower, upper = np.quantile(
            sampled_means,
            [0.025, 0.975],
        )

        wins = int(
            (differences > 0).sum()
        )

        losses = int(
            (differences < 0).sum()
        )

        ties = int(
            (differences == 0).sum()
        )

        decisive_gameweeks = (
            wins + losses
        )

        if decisive_gameweeks:
            win_rate = (
                wins
                / decisive_gameweeks
            )
        else:
            win_rate = np.nan

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "gameweeks": len(
                    differences
                ),
                "mean_xi_difference": float(
                    differences.mean()
                ),
                "median_xi_difference": float(
                    np.median(
                        differences
                    )
                ),
                "ci_95_lower": float(
                    lower
                ),
                "ci_95_upper": float(
                    upper
                ),
                "wins_a": wins,
                "losses_a": losses,
                "ties": ties,
                "win_rate_excluding_ties": (
                    win_rate
                ),
            }
        )

    return pd.DataFrame(rows)


def save_backtest_results(
    result: BacktestResult,
    output_dir: str | Path,
) -> None:
    """Save all backtest outputs as CSV files."""

    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.player_predictions.to_csv(
        output_path
        / "player_predictions.csv",
        index=False,
    )

    result.gameweek_results.to_csv(
        output_path
        / "gameweek_results.csv",
        index=False,
    )

    result.summary.to_csv(
        output_path
        / "summary.csv",
        index=False,
    )

    result.paired_comparisons.to_csv(
        output_path
        / "paired_comparisons.csv",
        index=False,
    )


def _position_points_per_90(
    prior: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate pooled historical points per 90 for each
    standard FPL position.
    """

    grouped = (
        prior
        .assign(
            position=(
                prior["position"]
                .astype(str)
                .str.upper()
            )
        )
        .loc[
            lambda frame: (
                frame["position"]
                .isin(
                    VALID_POSITIONS
                )
            )
        ]
        .groupby(
            "position",
            as_index=False,
        )
        .agg(
            points=(
                "total_points",
                "sum",
            ),
            minutes=(
                "minutes",
                "sum",
            ),
        )
    )

    grouped["points_per_90"] = np.where(
        grouped["minutes"] > 0,
        (
            90.0
            * grouped["points"]
            / grouped["minutes"]
        ),
        0.0,
    )

    return (
        grouped
        .set_index(
            "position"
        )["points_per_90"]
        .to_dict()
    )


def _team_fixture_weights(
    target_fixtures: pd.DataFrame,
    team_id_to_name: dict,
) -> dict[str, float]:
    """
    Calculate the total fixture weight for each team.

    Double-gameweek teams receive the sum of both fixture
    weights. Blank-gameweek teams retain a weight of zero.
    """

    weights: dict[str, float] = {}

    for _, fixture in (
        target_fixtures.iterrows()
    ):
        home_name = (
            team_id_to_name.get(
                int(
                    fixture["team_h"]
                )
            )
        )

        away_name = (
            team_id_to_name.get(
                int(
                    fixture["team_a"]
                )
            )
        )

        home_difficulty = int(
            fixture.get(
                "team_h_difficulty",
                3,
            )
        )

        away_difficulty = int(
            fixture.get(
                "team_a_difficulty",
                3,
            )
        )

        if home_name is not None:
            weights[home_name] = (
                weights.get(
                    home_name,
                    0.0,
                )
                + DIFF_TO_MULT.get(
                    home_difficulty,
                    1.0,
                )
            )

        if away_name is not None:
            weights[away_name] = (
                weights.get(
                    away_name,
                    0.0,
                )
                + DIFF_TO_MULT.get(
                    away_difficulty,
                    1.0,
                )
            )

    return weights


def _require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    filename: str,
) -> None:
    """Raise an error when an input file has an invalid schema."""

    missing = sorted(
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: "
            + ", ".join(missing)
        )