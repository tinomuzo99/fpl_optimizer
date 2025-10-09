import difflib
import pandas as pd

def _norm(s: str) -> str:
    return s.strip().lower().replace("-", " ").replace(".", "")

def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def _best_match(pool_df: pd.DataFrame, name: str, team: str | None = None, min_ratio: float = 0.8):
    """Return the best player match from pool_df based on name (and team if provided)."""
    name_n = _norm(name)
    team_n = _norm(team) if team else None

    best_row, best_score = None, 0.0

    for _, row in pool_df.iterrows():
        pool_name_n = _norm(row["name"])
        pool_team_n = _norm(row["team_name"])

        # base name similarity
        name_sim = _similar(name_n, pool_name_n)

        # team bonus if team matches closely
        if team_n:
            team_sim = _similar(team_n, pool_team_n)
            score = name_sim * 0.8 + team_sim * 0.2  # weighted combination
        else:
            score = name_sim

        if score > best_score:
            best_row, best_score = row, score

    if best_score >= min_ratio:
        return best_row
    else:
        return None


def map_names_to_pool(pool_df: pd.DataFrame, names_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map squad CSV names (and optionally teams) to the FPL pool.
    Accepts either:
      - one column: 'name'
      - or two columns: 'name', 'team'
    """
    if not {"name"}.issubset(names_df.columns):
        raise ValueError("Squad CSV must include at least a 'name' column.")

    if "team" not in names_df.columns:
        names_df["team"] = None  # fill if missing

    matched_rows = []
    for _, row in names_df.iterrows():
        nm, tm = str(row["name"]).strip(), str(row["team"]).strip() if row["team"] else None
        best = _best_match(pool_df, nm, tm)
        if best is None:
            print(f"[warn] Could not match: {nm}{' (' + tm + ')' if tm else ''}")
            continue
        matched_rows.append(best)

    if not matched_rows:
        raise ValueError("No valid player matches found from squad.csv — please check spelling or team names.")

    return pd.DataFrame(matched_rows).reset_index(drop=True)
