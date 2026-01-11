from __future__ import annotations

import os
from pathlib import Path
import json
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

from src.data.load import (
    load_train_labels, load_sample_submission,
    load_train_tracking, load_test_tracking
)
from src.features.parse import parse_contact_id
from src.features.build_features import build_pair_features


def main():
    print("CWD:", os.getcwd())

    # Output dirs
    reports_dir = Path("reports")
    models_dir = Path("models")
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_labels = load_train_labels()
    train_tracking = load_train_tracking()

    if "contact_id" in train_labels.columns and ("game_play" not in train_labels.columns or "step" not in train_labels.columns):
        train_labels = parse_contact_id(train_labels, col="contact_id")

    required = ["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"]
    if "contact" not in train_labels.columns:
        raise ValueError("Expected 'contact' column in train_labels.csv")

    # Features
    train_feat, feature_cols = build_pair_features(
        train_labels[required + ["contact"]],
        train_tracking
    )

    X = train_feat[feature_cols]
    y = train_feat["contact"].astype(int)
    groups = train_feat["game_play"].astype(str)

    # Group split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # LightGBM datasets
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbosity": -1,
        "seed": 42,
    }

    print("Training until validation scores don't improve for 100 rounds")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    )

    # Validate
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, val_pred)
    print(f"Validation ROC-AUC: {auc:.4f} | best_iteration={model.best_iteration}")

    # -------------------------
    # Save model + metadata
    # -------------------------
    model_path = models_dir / "lgbm_contact_model.txt"
    model.save_model(str(model_path))
    print("Saved model:", model_path)

    metadata = {
        "val_auc": float(auc),
        "best_iteration": int(model.best_iteration),
        "params": params,
        "feature_cols": feature_cols,
        "n_train_rows": int(len(train_feat)),
        "n_train_groups_game_play": int(train_feat["game_play"].nunique()),
    }
    meta_path = reports_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print("Saved metadata:", meta_path)

    # -------------------------
    # Feature importance
    # -------------------------
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    fi_path = reports_dir / "feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    print("Saved feature importance:", fi_path)
    print("Top 10 features (by gain):")
    print(fi.head(10).to_string(index=False))

    # -------------------------
    # Predict test + submission
    # -------------------------
    sample = load_sample_submission()
    test_tracking = load_test_tracking()

    test_pairs = parse_contact_id(sample[["contact_id"]], col="contact_id")
    test_feat, _ = build_pair_features(test_pairs[required], test_tracking)

    test_pred = model.predict(test_feat[feature_cols], num_iteration=model.best_iteration)

    sub = sample.copy()
    pred_col = "contact" if "contact" in sub.columns else sub.columns[-1]
    sub[pred_col] = test_pred

    sub_path = Path("submission.csv")
    sub.to_csv(sub_path, index=False)
    print("Wrote submission:", sub_path, sub.shape)


if __name__ == "__main__":
    main()