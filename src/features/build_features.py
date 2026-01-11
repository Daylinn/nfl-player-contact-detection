from __future__ import annotations

import numpy as np
import pandas as pd

# Candidate column names (in case Kaggle file schema differs slightly)
TRACKING_COL_CANDIDATES = {
    "x": ["x", "x_position"],
    "y": ["y", "y_position"],
    "s": ["s", "speed"],
    "a": ["a", "acceleration"],
    "dir": ["dir", "direction"],
    "o": ["o", "orientation"],
    "nfl_player_id": ["nfl_player_id", "player_id"],
    "game_play": ["game_play"],
    "step": ["step", "frame"],
}


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Pick the first existing column from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def standardize_tracking_cols(tracking: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize tracking dataframe columns to a consistent schema:
      game_play, step, nfl_player_id, (optional) x,y,s,a,dir,o
    """
    t = tracking.copy()

    gp = _pick_col(t, TRACKING_COL_CANDIDATES["game_play"])
    st = _pick_col(t, TRACKING_COL_CANDIDATES["step"])
    pid = _pick_col(t, TRACKING_COL_CANDIDATES["nfl_player_id"])

    rename = {gp: "game_play", st: "step", pid: "nfl_player_id"}

    # Optional numeric cols if present
    for k in ["x", "y", "s", "a", "dir", "o"]:
        for cand in TRACKING_COL_CANDIDATES[k]:
            if cand in t.columns:
                rename[cand] = k
                break

    t = t.rename(columns=rename)

    # Types
    t["step"] = pd.to_numeric(t["step"], errors="coerce")
    t["nfl_player_id"] = pd.to_numeric(t["nfl_player_id"], errors="coerce")

    return t


def build_pair_features(pairs: pd.DataFrame, tracking: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build features for each (game_play, step, nfl_player_id_1, nfl_player_id_2) row.

    pairs must contain:
      - game_play, step, nfl_player_id_1, nfl_player_id_2

    tracking should contain:
      - game_play, step, nfl_player_id, plus optional columns x, y, s, a, dir, o

    Returns:
      - df with merged tracking info for player 1 and player 2 plus engineered features
      - list of feature column names
    """
    t = standardize_tracking_cols(tracking)

    # Create p1 and p2 views of tracking (suffix columns)
    t1 = t.rename(columns={c: f"{c}_1" for c in t.columns if c not in ["game_play", "step"]})
    t2 = t.rename(columns={c: f"{c}_2" for c in t.columns if c not in ["game_play", "step"]})

    df = pairs.copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["nfl_player_id_1"] = pd.to_numeric(df["nfl_player_id_1"], errors="coerce")
    df["nfl_player_id_2"] = pd.to_numeric(df["nfl_player_id_2"], errors="coerce")

    # Merge player 1 tracking
    df = df.merge(
        t1,
        how="left",
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id_1"],
    )

    # Merge player 2 tracking
    df = df.merge(
        t2,
        how="left",
        left_on=["game_play", "step", "nfl_player_id_2"],
        right_on=["game_play", "step", "nfl_player_id_2"],
    )

    # -----------------------------
    # Geometric features (distance)
    # -----------------------------
    if {"x_1", "y_1", "x_2", "y_2"}.issubset(df.columns):
        df["dx"] = df["x_1"] - df["x_2"]
        df["dy"] = df["y_1"] - df["y_2"]
        df["dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    else:
        df["dx"] = np.nan
        df["dy"] = np.nan
        df["dist"] = np.nan

    # -----------------------------
    # Simple motion features
    # -----------------------------
    if {"s_1", "s_2"}.issubset(df.columns):
        df["rel_speed"] = (df["s_1"] - df["s_2"]).abs()
        df["speed_sum"] = df["s_1"].fillna(0) + df["s_2"].fillna(0)
    else:
        df["rel_speed"] = np.nan
        df["speed_sum"] = np.nan

    if {"a_1", "a_2"}.issubset(df.columns):
        df["rel_acc"] = (df["a_1"] - df["a_2"]).abs()
    else:
        df["rel_acc"] = np.nan

    # Angle difference helper (degrees)
    def ang_diff(a: pd.Series, b: pd.Series) -> np.ndarray:
        d = (a - b) % 360
        return np.minimum(d, 360 - d)

    if {"dir_1", "dir_2"}.issubset(df.columns):
        df["dir_diff"] = ang_diff(df["dir_1"], df["dir_2"])
    else:
        df["dir_diff"] = np.nan

    # -----------------------------------------
    # Physics features: velocity + closing speed
    # -----------------------------------------
    # closing_speed > 0 means players are moving toward each other along the line between them
    if {"s_1", "s_2", "dir_1", "dir_2", "dx", "dy", "dist"}.issubset(df.columns):
        dir1 = np.deg2rad(df["dir_1"])
        dir2 = np.deg2rad(df["dir_2"])

        # velocity components (approx)
        df["vx_1"] = df["s_1"] * np.cos(dir1)
        df["vy_1"] = df["s_1"] * np.sin(dir1)
        df["vx_2"] = df["s_2"] * np.cos(dir2)
        df["vy_2"] = df["s_2"] * np.sin(dir2)

        # relative velocity (1 - 2)
        df["rvx"] = df["vx_1"] - df["vx_2"]
        df["rvy"] = df["vy_1"] - df["vy_2"]

        # closing speed: -(rel_v dot rel_pos_unit)
        eps = 1e-6
        df["closing_speed"] = - (df["rvx"] * df["dx"] + df["rvy"] * df["dy"]) / (df["dist"] + eps)

        # relative velocity magnitude
        df["rel_speed_vec"] = np.sqrt(df["rvx"] ** 2 + df["rvy"] ** 2)
    else:
        df["vx_1"] = np.nan
        df["vy_1"] = np.nan
        df["vx_2"] = np.nan
        df["vy_2"] = np.nan
        df["rvx"] = np.nan
        df["rvy"] = np.nan
        df["closing_speed"] = np.nan
        df["rel_speed_vec"] = np.nan

    feature_cols = [
        "dist", "dx", "dy",
        "rel_speed", "speed_sum", "rel_acc", "dir_diff",
        "closing_speed", "rel_speed_vec",
    ]

    return df, feature_cols