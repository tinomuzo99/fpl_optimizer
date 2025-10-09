from __future__ import annotations
from typing import Dict, Optional, Set, Literal
import pandas as pd
import pulp

from .config import REQUIRED_COUNTS, MAX_PER_TEAM, DIFF_TO_MULT, AVAIL_SCALE
from .fpl import fetch_bootstrap, fetch_fixtures, build_player_table, add_horizon_expected_points


# ---------------------------------------------------------------------------
# Build the player pool with BOTH metrics:
#   - exp_points_next: next GW (from ep_next/ppg fallback)
#   - exp_points_h:    weighted sum over the horizon window
# Also produce availability-adjusted versions for optimisation:
#   - exp_points_next_avail
#   - exp_points_h_avail
# ---------------------------------------------------------------------------
def prepare_player_pool(allow_flagged: bool, min_play_chance: int, horizon: int) -> pd.DataFrame:
    bootstrap = fetch_bootstrap()
    fixtures_df = fetch_fixtures()

    pool0 = build_player_table(bootstrap, allow_flagged=allow_flagged, min_play_chance=min_play_chance)
    pool_h = add_horizon_expected_points(pool0, bootstrap, fixtures_df, horizon=horizon, diff_to_mult=DIFF_TO_MULT)

    # Attach status and availability multipliers
    pool_h = pool_h.merge(
        pd.DataFrame(bootstrap["elements"])[["id", "status"]],
        on="id", how="left"
    )
    pool_h["avail_mult"] = pool_h["status"].map(AVAIL_SCALE).fillna(0.8)

    # Availability-adjusted metrics
    pool_h["exp_points_next_avail"] = pool_h["exp_points_next"] * pool_h["avail_mult"]
    pool_h["exp_points_h_avail"]    = pool_h["exp_points_h"]    * pool_h["avail_mult"]

    return pool_h


def _score_col(score_type: Literal["next", "horizon"]) -> str:
    return "exp_points_next_avail" if score_type == "next" else "exp_points_h_avail"


# ---------------------------------------------------------------------------
# Optimise best 15 given budget/constraints and a chosen score metric
# ---------------------------------------------------------------------------
def optimise_squad(
    players: pd.DataFrame,
    budget_m: float,
    score_type: Literal["next", "horizon"] = "horizon",
    max_per_team: int = MAX_PER_TEAM,
    required_counts: Dict[str, int] = REQUIRED_COUNTS,
    lock_names: Optional[Set[str]] = None,
    ban_names: Optional[Set[str]] = None,
    min_from_team: Optional[Dict[str, int]] = None,
    verbose: bool = False,
):
    score = _score_col(score_type)

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


# ---------------------------------------------------------------------------
# Starting XI using chosen metric
# ---------------------------------------------------------------------------
def choose_starting_xi(squad_df: pd.DataFrame, score_type: Literal["next", "horizon"] = "horizon"):
    score = _score_col(score_type)
    df = squad_df.copy()

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
    score_type: Literal["next", "horizon"] = "horizon",
) -> pd.DataFrame:
    """
    Evaluate best single-transfer upgrades using the provided score_type.

    pool_df / current_df must contain:
      - 'name', 'team_name', 'pos_code', 'cost_m'
      - 'exp_points_next_avail', 'exp_points_h_avail'
    """
    score = _score_col(score_type)

    base_xi_pts = choose_starting_xi(current_df, score_type)[2]
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
