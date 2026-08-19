from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import fpl_optimizer.optimizer as optimizer_module


SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def load_snapshot(filename: str):
    path = SNAPSHOT_DIR / filename

    if not path.exists():
        pytest.fail(
            f"Missing API snapshot: {path}. "
            "Run `python scripts/update_api_snapshots.py` first."
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="session")
def bootstrap_snapshot() -> dict:
    return load_snapshot("bootstrap_static.json")


@pytest.fixture(scope="session")
def fixtures_snapshot() -> list[dict]:
    return load_snapshot("fixtures.json")


@pytest.fixture
def mocked_fpl_api(
    monkeypatch,
    bootstrap_snapshot,
    fixtures_snapshot,
):
    """
    Replace live FPL API calls with committed snapshot data.
    """

    monkeypatch.setattr(
        optimizer_module,
        "fetch_bootstrap",
        lambda: copy.deepcopy(bootstrap_snapshot),
    )

    monkeypatch.setattr(
        optimizer_module,
        "fetch_fixtures",
        lambda: pd.DataFrame(
            copy.deepcopy(fixtures_snapshot)
        ),
    )

    return {
        "bootstrap": bootstrap_snapshot,
        "fixtures": fixtures_snapshot,
    }