from __future__ import annotations

import pandas as pd
import pytest

from fpl_optimizer.config import (
    DIFF_TO_MULT,
    MAX_PER_TEAM,
    REQUIRED_COUNTS,
)
from fpl_optimizer.fpl import (
    add_horizon_expected_points,
    build_player_table,
    compute_team_gw_weights,
    get_next_gw_and_window,
)
from fpl_optimizer.optimizer import (
    choose_starting_xi,
    optimise_squad,
    prepare_player_pool,
    validate_squad,
)


def test_snapshot_contains_required_sections(
    bootstrap_snapshot,
):
    assert "events" in bootstrap_snapshot
    assert "teams" in bootstrap_snapshot
    assert "elements" in bootstrap_snapshot

    assert bootstrap_snapshot["events"]
    assert bootstrap_snapshot["teams"]
    assert bootstrap_snapshot["elements"]


def test_snapshot_player_schema(
    bootstrap_snapshot,
):
    required_fields = {
        "id",
        "web_name",
        "team",
        "element_type",
        "now_cost",
        "ep_next",
        "points_per_game",
        "status",
        "chance_of_playing_next_round",
    }

    for player in bootstrap_snapshot["elements"]:
        missing = required_fields - set(player)

        assert not missing, (
            f"Player {player.get('id')} is missing "
            f"fields: {sorted(missing)}"
        )


def test_next_gameweek_window_uses_snapshot(
    bootstrap_snapshot,
):
    next_gameweek, window = get_next_gw_and_window(
        bootstrap_snapshot,
        horizon=4,
    )

    assert isinstance(next_gameweek, int)
    assert next_gameweek >= 1
    assert window == [
        next_gameweek,
        next_gameweek + 1,
        next_gameweek + 2,
        next_gameweek + 3,
    ]


def test_build_player_table_from_snapshot(
    bootstrap_snapshot,
):
    players = build_player_table(
        bootstrap_snapshot,
        allow_flagged=True,
        min_play_chance=0,
    )

    assert not players.empty

    expected_columns = {
        "id",
        "name",
        "team_id",
        "team_name",
        "pos_code",
        "cost_m",
        "ppg",
        "exp_points_next",
    }

    assert expected_columns.issubset(players.columns)

    assert players["id"].is_unique
    assert players["name"].notna().all()
    assert players["team_name"].notna().all()

    assert set(players["pos_code"].unique()).issubset(
        {"GK", "DEF", "MID", "FWD"}
    )

    assert (players["cost_m"] > 0).all()
    assert (players["exp_points_next"] >= 0).all()


def test_player_cost_conversion(
    bootstrap_snapshot,
):
    first_api_player = bootstrap_snapshot["elements"][0]

    players = build_player_table(
        bootstrap_snapshot,
        allow_flagged=True,
        min_play_chance=0,
    )

    transformed_player = players.loc[
        players["id"] == first_api_player["id"]
    ].iloc[0]

    assert transformed_player["cost_m"] == pytest.approx(
        first_api_player["now_cost"] / 10.0
    )


def test_fixture_weights_cover_every_team_and_gameweek(
    bootstrap_snapshot,
    fixtures_snapshot,
):
    fixtures_df = pd.DataFrame(fixtures_snapshot)

    _, window = get_next_gw_and_window(
        bootstrap_snapshot,
        horizon=4,
    )

    weights = compute_team_gw_weights(
        bootstrap_snapshot,
        fixtures_df,
        window,
        DIFF_TO_MULT,
    )

    team_ids = {
        int(team["id"])
        for team in bootstrap_snapshot["teams"]
    }

    for team_id in team_ids:
        for gameweek in window:
            assert (team_id, gameweek) in weights
            assert weights[(team_id, gameweek)] >= 0


def test_horizon_points_from_snapshot(
    bootstrap_snapshot,
    fixtures_snapshot,
):
    fixtures_df = pd.DataFrame(fixtures_snapshot)

    players = build_player_table(
        bootstrap_snapshot,
        allow_flagged=True,
        min_play_chance=0,
    )

    horizon_players = add_horizon_expected_points(
        players,
        bootstrap_snapshot,
        fixtures_df,
        horizon=4,
        diff_to_mult=DIFF_TO_MULT,
    )

    assert "exp_points_h" in horizon_players.columns
    assert horizon_players["exp_points_h"].notna().all()
    assert (horizon_players["exp_points_h"] >= 0).all()


def test_prepare_player_pool_without_live_network(
    mocked_fpl_api,
):
    pool = prepare_player_pool(
        allow_flagged=True,
        min_play_chance=0,
        horizon=4,
    )

    expected_columns = {
        "id",
        "name",
        "team_name",
        "pos_code",
        "cost_m",
        "status",
        "avail_mult",
        "exp_points_next_avail",
        "exp_points_h_avail",
    }

    assert not pool.empty
    assert expected_columns.issubset(pool.columns)

    assert pool["id"].is_unique
    assert pool["exp_points_next_avail"].notna().all()
    assert pool["exp_points_h_avail"].notna().all()


def test_snapshot_produces_legal_optimised_squad(
    mocked_fpl_api,
):
    pool = prepare_player_pool(
        allow_flagged=True,
        min_play_chance=0,
        horizon=4,
    )

    (
        strategy,
        squad,
        total_cost,
        total_points,
        score_column,
    ) = optimise_squad(
        pool,
        budget_m=100.0,
        score_type="horizon",
    )

    assert strategy == "strict"
    assert score_column == "exp_points_h_avail"

    validate_squad(squad)

    assert len(squad) == 15
    assert total_cost <= 100.0
    assert total_points >= 0

    assert squad["pos_code"].value_counts().to_dict() == {
        "DEF": REQUIRED_COUNTS["DEF"],
        "MID": REQUIRED_COUNTS["MID"],
        "FWD": REQUIRED_COUNTS["FWD"],
        "GK": REQUIRED_COUNTS["GK"],
    }

    assert (
        squad["team_name"].value_counts().max()
        <= MAX_PER_TEAM
    )


def test_snapshot_squad_produces_legal_starting_xi(
    mocked_fpl_api,
):
    pool = prepare_player_pool(
        allow_flagged=True,
        min_play_chance=0,
        horizon=4,
    )

    _, squad, _, _, _ = optimise_squad(
        pool,
        budget_m=100.0,
        score_type="horizon",
    )

    starting_xi, bench, expected_points = (
        choose_starting_xi(
            squad,
            score_type="horizon",
        )
    )

    position_counts = (
        starting_xi["pos_code"]
        .value_counts()
        .to_dict()
    )

    assert len(starting_xi) == 11
    assert len(bench) == 4

    assert position_counts.get("GK", 0) == 1
    assert position_counts.get("DEF", 0) >= 3
    assert position_counts.get("MID", 0) >= 2
    assert position_counts.get("FWD", 0) >= 1

    assert expected_points >= 0

def test_snapshot_tests_do_not_call_live_api(
    monkeypatch,
    mocked_fpl_api,
):
    import requests

    def reject_network_call(*args, **kwargs):
        raise AssertionError(
            "A live network request was attempted during testing."
        )

    monkeypatch.setattr(
        requests,
        "get",
        reject_network_call,
    )

    pool = prepare_player_pool(
        allow_flagged=True,
        min_play_chance=0,
        horizon=4,
    )

    assert not pool.empty


    