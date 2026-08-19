from __future__ import annotations

import argparse
from pathlib import Path

import requests


BASE_URL = (
    "https://raw.githubusercontent.com/vaastav/"
    "Fantasy-Premier-League/master/data"
)
FILES = {
    "merged_gw.csv": "gws/merged_gw.csv",
    "fixtures.csv": "fixtures.csv",
    "teams.csv": "teams.csv",
}


def download(url: str, destination: Path) -> None:
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "fpl-optimizer-backtest/1.0"},
    )
    response.raise_for_status()
    destination.write_bytes(response.content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one archived FPL season for backtesting."
    )
    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--out-dir", default="data/historical")
    args = parser.parse_args()

    season_dir = Path(args.out_dir) / args.season
    season_dir.mkdir(parents=True, exist_ok=True)

    for output_name, remote_path in FILES.items():
        url = f"{BASE_URL}/{args.season}/{remote_path}"
        destination = season_dir / output_name
        print(f"Fetching {url}")
        download(url, destination)
        print(f"Saved {destination}")

    print("Historical data download complete.")


if __name__ == "__main__":
    main()
