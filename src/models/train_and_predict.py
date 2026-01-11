from __future__ import annotations
import os
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

    train_labels = load_train_labels()
    train_tracking = load_train_tracking()

    if "contact_id" in train_labels.columns and ("game_play" not in train_labels.columns or "step" not in train_labels.columns):
        train_labels = parse_contact_id(train_labels, col="contact_id")

    required = ["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"]
    if "contact" not in train_labels.columns:
        raise ValueError("Expected 'contact' column in train_labels.csv")

    train_feat, feature_cols = build_pair_features(train_labels[required + ["contact"]], train_tracking)

    X = train_feat[feature_cols]
    y = train_feat["contact"].astype(int)
    groups = train_feat["game_play"].astype(str)

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

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=50)],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, val_pred)
    print(f"Validation ROC-AUC: {auc:.4f} | best_iteration={model.best_iteration}")

    # Predict test + write submission
    sample = load_sample_submission()
    test_tracking = load_test_tracking()

    test_pairs = parse_contact_id(sample[["contact_id"]], col="contact_id")
    test_feat, _ = build_pair_features(test_pairs[required], test_tracking)

    test_pred = model.predict(test_feat[feature_cols], num_iteration=model.best_iteration)

    sub = sample.copy()
    pred_col = "contact" if "contact" in sub.columns else sub.columns[-1]
    sub[pred_col] = test_pred

    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv", sub.shape)


if __name__ == "__main__":
    main()