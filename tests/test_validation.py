import pandas as pd
import pytest

from fpl_optimizer.cli import read_squad_csv
from fpl_optimizer.name_matching import map_names_to_pool
from fpl_optimizer.optimizer import (
    choose_starting_xi,
    optimise_squad,
    optimise_squad_with_xi,
    validate_squad,
)
def valid_squad() -> pd.DataFrame:
    positions = (
        ["GK"] * 2
        + ["DEF"] * 5
        + ["MID"] * 5
        + ["FWD"] * 3
    )

    teams = [
        "A", "A", "A",
        "B", "B", "B",
        "C", "C", "C",
        "D", "D", "D",
        "E", "E", "E",
    ]

    return pd.DataFrame(
        {
            "id": range(1, 16),
            "name": [
                f"Player {i}"
                for i in range(1, 16)
            ],
            "team_name": teams,
            "pos_code": positions,
            "cost_m": [5.0] * 15,
            "exp_points_next_avail": [
                float(i)
                for i in range(1, 16)
            ],
            "exp_points_h_avail": [
                float(i)
                for i in range(1, 16)
            ],
        }
    )


def test_joint_optimizer_returns_legal_xi_and_captain():
    plan = optimise_squad_with_xi(
        valid_squad(),
        budget_m=100.0,
        score_type="next",
        bench_weight=0.1,
    )

    validate_squad(plan.squad)

    assert len(plan.starting_xi) == 11
    assert len(plan.bench) == 4

    assert plan.captain["id"] in set(
        plan.starting_xi["id"]
    )

    assert plan.vice_captain["id"] in set(
        plan.starting_xi["id"]
    )

    assert (
        plan.captain["id"]
        != plan.vice_captain["id"]
    )

    positions = (
        plan.starting_xi["pos_code"]
        .value_counts()
        .to_dict()
    )

    assert positions.get("GK", 0) == 1
    assert positions.get("DEF", 0) >= 3
    assert positions.get("MID", 0) >= 2
    assert positions.get("FWD", 0) >= 1

    assert plan.predicted_fpl_points == pytest.approx(
        plan.predicted_xi_points
        + plan.predicted_captain_bonus
    )


def test_joint_optimizer_rejects_invalid_bench_weight():
    with pytest.raises(
        ValueError,
        match="bench_weight must be between 0 and 1",
    ):
        optimise_squad_with_xi(
            valid_squad(),
            budget_m=100.0,
            bench_weight=1.1,
        )