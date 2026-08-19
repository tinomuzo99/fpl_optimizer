import pandas as pd
import pytest

from fpl_optimizer.backtest import (
    build_gameweek_pool,
    calculate_prediction_metrics,
    run_historical_backtest,
)


def historical_frames():
    teams = pd.DataFrame(
        {"id": range(1, 7), "name": [f"Team {i}" for i in range(1, 7)]}
    )
    fixtures = []
    for gw in (1, 2, 3):
        for home, away in ((1, 2), (3, 4), (5, 6)):
            fixtures.append(
                {
                    "event": gw,
                    "team_h": home,
                    "team_a": away,
                    "team_h_difficulty": 2,
                    "team_a_difficulty": 4,
                }
            )

    positions = ["GK", "DEF", "DEF", "MID", "MID", "FWD"]
    rows = []
    element = 1
    for team_id in range(1, 7):
        for position in positions:
            for gw in (1, 2, 3):
                rows.append(
                    {
                        "element": element,
                        "name": f"Player {element}",
                        "position": position,
                        "team": f"Team {team_id}",
                        "total_points": 100 if (element == 1 and gw == 3) else 2,
                        "minutes": 90,
                        "value": 45 + (element % 5),
                        "GW": gw,
                        "kickoff_time": f"2024-09-{gw:02d}T12:00:00Z",
                        "xP": 999 if gw == 3 else 1,
                    }
                )
            element += 1

    return pd.DataFrame(rows), pd.DataFrame(fixtures), teams


def test_target_gameweek_points_and_xp_do_not_enter_prediction():
    gameweeks, fixtures, teams = historical_frames()
    pool = build_gameweek_pool(
        gameweeks, fixtures, teams, target_gw=3, model="prior_ppg"
    )
    player = pool.loc[pool["id"] == 1].iloc[0]

    assert player["prior_points"] == 4
    assert player["prior_appearances"] == 2
    assert player["exp_points_next"] == pytest.approx(2.0)
    assert player["actual_points"] == 100


def test_double_gameweek_actual_points_are_summed():
    gameweeks, fixtures, teams = historical_frames()
    extra_fixture = fixtures.iloc[[0]].copy()
    extra_fixture["event"] = 3
    fixtures = pd.concat([fixtures, extra_fixture], ignore_index=True)
    extra_player_row = gameweeks.loc[
        (gameweeks["element"] == 1) & (gameweeks["GW"] == 3)
    ].copy()
    extra_player_row["total_points"] = 5
    extra_player_row["kickoff_time"] = "2024-09-04T12:00:00Z"
    gameweeks = pd.concat([gameweeks, extra_player_row], ignore_index=True)

    pool = build_gameweek_pool(gameweeks, fixtures, teams, 3, "fixture_ppg")

    assert pool.loc[pool["id"] == 1, "actual_points"].iloc[0] == 105

def test_backtest_produces_all_model_results():
    gameweeks, fixtures, teams = historical_frames()

    result = run_historical_backtest(
        gameweeks,
        fixtures,
        teams,
        gameweek_range=[3],
        budget_m=100.0,
        top_k=10,
    )

    assert set(result.gameweek_results["model"]) == {
        "prior_ppg",
        "fixture_ppg",
        "minutes_shrunk_fixture",
    }

    assert len(result.summary) == 3
    assert len(result.paired_comparisons) == 3

    selected_squads = (
        result.player_predictions
        .groupby(["GW", "model"])["selected_squad"]
        .sum()
    )

    selected_elevens = (
        result.player_predictions
        .groupby(["GW", "model"])["selected_xi"]
        .sum()
    )

    assert selected_squads.eq(15).all()
    assert selected_elevens.eq(11).all()


def test_minutes_model_uses_only_prior_minutes_and_shrinkage():
    gameweeks, fixtures, teams = historical_frames()

    player_mask = gameweeks["element"] == 1

    gameweeks.loc[
        player_mask & (gameweeks["GW"] == 1),
        "minutes",
    ] = 90

    gameweeks.loc[
        player_mask & (gameweeks["GW"] == 2),
        "minutes",
    ] = 0

    pool = build_gameweek_pool(
        gameweeks,
        fixtures,
        teams,
        target_gw=3,
        model="minutes_shrunk_fixture",
        recent_gws=2,
        shrinkage_matches=5,
    )

    player = pool.loc[
        pool["id"] == 1
    ].iloc[0]

    assert player["recent_minutes"] == 90
    assert player["recent_fixture_rows"] == 2
    assert player["expected_minutes_share"] == pytest.approx(0.5)


def test_prediction_metrics_are_finite():
    pool = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "exp_points_next_avail": [3.0, 2.0, 1.0],
            "actual_points": [2.0, 3.0, 0.0],
        }
    )
    metrics = calculate_prediction_metrics(pool, top_k=2)

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["top_k_recall"] == pytest.approx(1.0)
