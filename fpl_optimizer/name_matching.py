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


def map_names_to_pool(
    pool_df: pd.DataFrame,
    names_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map squad CSV names and optional teams to the current FPL pool.

    Every supplied player must be matched. The function raises an
    error rather than returning an incomplete squad.
    """

    if "name" not in names_df.columns:
        raise ValueError(
            "Squad CSV must include at least a 'name' column."
        )

    names_df = names_df.copy()

    if "team" not in names_df.columns:
        names_df["team"] = None

    matched_rows = []
    unmatched = []

    for _, row in names_df.iterrows():
        name = str(row["name"]).strip()

        team = (
            str(row["team"]).strip()
            if row["team"]
            else None
        )

        best_match = _best_match(
            pool_df,
            name,
            team,
        )

        if best_match is None:
            label = (
                f"{name} ({team})"
                if team
                else name
            )
            unmatched.append(label)
            continue

        matched_rows.append(best_match)

    if unmatched:
        raise ValueError(
            "Could not match all squad players against the "
            "current FPL data: "
            + ", ".join(unmatched)
            + ". Update the names or teams in the squad file "
              "and try again."
        )

    return pd.DataFrame(
        matched_rows
    ).reset_index(drop=True)
