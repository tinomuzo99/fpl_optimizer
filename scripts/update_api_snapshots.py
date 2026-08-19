from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = PROJECT_ROOT / "tests" / "snapshots"

FPL_API_BASE = "https://fantasy.premierleague.com/api"

ENDPOINTS = {
    "bootstrap_static.json": f"{FPL_API_BASE}/bootstrap-static/",
    "fixtures.json": f"{FPL_API_BASE}/fixtures/",
}


def fetch_json(url: str):
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "fpl-optimizer-snapshot-updater/1.0",
        },
    )
    response.raise_for_status()
    return response.json()


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in ENDPOINTS.items():
        print(f"Fetching {url}")
        data = fetch_json(url)

        output_path = SNAPSHOT_DIR / filename
        write_json(output_path, data)

        print(f"Saved {output_path}")

    metadata = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": FPL_API_BASE,
        "files": list(ENDPOINTS),
    }

    write_json(
        SNAPSHOT_DIR / "metadata.json",
        metadata,
    )

    print("Snapshot update complete.")


if __name__ == "__main__":
    main()