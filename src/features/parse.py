from __future__ import annotations
import pandas as pd

def parse_contact_id(df: pd.DataFrame, col: str = "contact_id") -> pd.DataFrame:
    """
    Expected contact_id pattern usually like:
      {game_play}_{step}_{nfl_player_id_1}_{nfl_player_id_2}
    Some competitions may include additional tokens.
    We'll parse from the right side to be safe:
      last 3 tokens = step, p1, p2
      everything before that = game_play
    """
    s = df[col].astype(str)

    parts = s.str.split("_")

    # last 3 tokens
    step = parts.str[-3]
    p1 = parts.str[-2]
    p2 = parts.str[-1]

    # game_play is everything before last 3
    game_play = parts.apply(lambda x: "_".join(x[:-3]) if len(x) > 3 else None)

    out = df.copy()
    out["game_play"] = game_play
    out["step"] = pd.to_numeric(step, errors="coerce")
    out["nfl_player_id_1"] = p1
    out["nfl_player_id_2"] = p2

    # Convert player ids to numeric where possible; keep NaN if not numeric
    out["nfl_player_id_1"] = pd.to_numeric(out["nfl_player_id_1"], errors="coerce")
    out["nfl_player_id_2"] = pd.to_numeric(out["nfl_player_id_2"], errors="coerce")

    return out