from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import requests
import pandas as pd

from .config import BOOTSTRAP_URL, FIXTURES_URL

def fetch_bootstrap() -> dict:
    r = requests.get(BOOTSTRAP_URL, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_fixtures() -> pd.DataFrame:
    r = requests.get(FIXTURES_URL, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def get_next_gw_and_window(bootstrap: dict, horizon: int) -> Tuple[int, List[int]]:
    events = pd.DataFrame(bootstrap["events"])
    if "is_next" in events.columns and events["is_next"].any():
        next_gw = int(events.loc[events["is_next"], "id"].iloc[0])
    else:
        unfinished = events.loc[~events.get("finished", False), "id"]
        next_gw = int(unfinished.min()) if len(unfinished) else int(events["id"].max())
    window = list(range(next_gw, next_gw + horizon))
    return next_gw, window

def build_player_table(bootstrap: dict, allow_flagged: bool = False, min_play_chance: Optional[int] = None) -> pd.DataFrame:
    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])

    team_id_to_name = teams.set_index("id")["name"].to_dict()
    id_to_pos = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    df = elements[[
        "id","web_name","team","element_type","now_cost",
        "ep_next","points_per_game","status","chance_of_playing_next_round"
    ]].copy()

    df["team_id"]   = df["team"]
    df["team_name"] = df["team_id"].map(team_id_to_name)
    df["pos_code"]  = df["element_type"].map(id_to_pos)
    df.rename(columns={"web_name":"name"}, inplace=True)

    def to_float(x):
        try: return float(x)
        except Exception: return math.nan

    df["ep_next"] = df["ep_next"].apply(to_float)
    df["ppg"] = pd.to_numeric(df["points_per_game"], errors="coerce")
    df["exp_points_next"] = df["ep_next"].fillna(df["ppg"]).fillna(0.0)
    df["cost_m"] = df["now_cost"] / 10.0

    if not allow_flagged:
        thr = 75 if min_play_chance is None else int(min_play_chance)
        ch = df["chance_of_playing_next_round"].fillna(100)
        df = df[(df["status"] == "a") | (ch >= thr)]

    keep = ["id","name","team_id","team_name","pos_code","cost_m","ppg","exp_points_next"]
    return df[keep].reset_index(drop=True)

def compute_team_gw_weights(bootstrap: dict, fixtures_df: pd.DataFrame,
                            gw_window: List[int], diff_to_mult: Dict[int, float]) -> Dict[Tuple[int,int], float]:
    weights = {}
    fx = fixtures_df[fixtures_df["event"].isin(gw_window)].copy()
    fx = fx.dropna(subset=["event"])
    fx["event"] = fx["event"].astype(int)

    for _, row in fx.iterrows():
        gw = int(row["event"])
        t_h = int(row["team_h"]); diff_h = int(row.get("team_h_difficulty", 3))
        t_a = int(row["team_a"]); diff_a = int(row.get("team_a_difficulty", 3))
        w_h = diff_to_mult.get(diff_h, 1.0)
        w_a = diff_to_mult.get(diff_a, 1.0)
        weights[(t_h, gw)] = weights.get((t_h, gw), 0.0) + w_h
        weights[(t_a, gw)] = weights.get((t_a, gw), 0.0) + w_a

    teams_df = pd.DataFrame(bootstrap["teams"])
    for team_id in teams_df["id"].tolist():
        for gw in gw_window:
            weights.setdefault((int(team_id), int(gw)), 0.0)
    return weights

def add_horizon_expected_points(players_df: pd.DataFrame,
                                bootstrap: dict,
                                fixtures_df: pd.DataFrame,
                                horizon: int,
                                diff_to_mult: Dict[int, float]) -> pd.DataFrame:
    next_gw, window = get_next_gw_and_window(bootstrap, horizon)
    weights = compute_team_gw_weights(bootstrap, fixtures_df, window, diff_to_mult)

    def horizon_sum(row):
        team_id = int(row["team_id"])
        base = float(row["ppg"]) if pd.notnull(row["ppg"]) else (
               float(row["exp_points_next"]) if pd.notnull(row["exp_points_next"]) else 0.0)
        total = 0.0
        for gw in window:
            w = float(weights.get((team_id, int(gw)), 0.0))
            total += base * w
        return total

    df = players_df.copy()
    df["exp_points_h"] = df.apply(horizon_sum, axis=1)
    df.attrs["next_gw"] = next_gw
    df.attrs["gw_window"] = window
    return df
