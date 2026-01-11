# NFL Player Contact Detection

Binary classification of player-to-player contact events from NFL tracking data using physics-informed features and gradient boosting.

---

## Problem Overview

Predict whether two players make contact at a given game frame using only tracking sensor data (position, velocity, acceleration, orientation). The target is imbalanced binary labels (contact/no-contact) over ~4.7M player-pair observations across 240 game plays.

**Key challenge**: Distinguish true physical contact from near-misses using spatial and kinematic signals.

---

## Approach

**Baseline**: LightGBM classifier on engineered tracking features, no helmet/video data.

### Feature Engineering

All features derived from per-frame tracking data for each (player_1, player_2) pair:

**Geometric**
- `dist` – Euclidean distance between players
- `dx`, `dy` – Component-wise separation

**Motion**
- `rel_speed` – Absolute speed difference
- `speed_sum` – Combined speeds
- `rel_acc` – Absolute acceleration difference
- `dir_diff` – Angular difference in heading (degrees)

**Physics-Informed**
- `closing_speed` – Rate of approach along the line connecting players (negative of relative velocity projected onto separation vector)
- `rel_speed_vec` – Magnitude of relative velocity

Implementation: `src/features/build_features.py:57`

---

## Repository Structure

```
.
├── src/
│   ├── features/
│   │   ├── build_features.py    # Feature engineering pipeline
│   │   └── parse.py             # Contact ID parser
│   ├── models/
│   │   ├── train_and_predict.py      # Baseline LightGBM training
│   │   └── train_and_predict_lgbm.py # Full training + metadata
│   └── utils/
│       └── paths.py
├── reports/
│   ├── run_metadata.json        # Validation metrics, hyperparams
│   └── feature_importance.csv   # Gain/split importance (if generated)
├── models/
│   └── lgbm_contact_model.txt   # Trained LightGBM (if generated)
└── submission.csv               # Kaggle-format predictions
```

**Note**: The `src/data/load.py` module (for loading CSVs) is referenced by training scripts but not committed. You'll need to implement loaders for:
- `load_train_labels()` → train labels CSV
- `load_train_tracking()` → train tracking CSV
- `load_test_tracking()` → test tracking CSV
- `load_sample_submission()` → sample submission CSV

Kaggle dataset files should be stored in `data/` (gitignored).

---

## Setup

### Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm
```

### macOS: Install OpenMP

LightGBM requires OpenMP on macOS:

```bash
brew install libomp
```

---

## How to Run

### 1. Baseline Training + Submission

```bash
python -m src.models.train_and_predict
```

**Outputs**: `submission.csv`

### 2. Full Training with Artifacts

```bash
python -m src.models.train_and_predict_lgbm
```

**Outputs**:
- `submission.csv` (49,588 rows)
- `reports/run_metadata.json`
- `reports/feature_importance.csv`
- `models/lgbm_contact_model.txt`

---

## Validation Strategy

**Split**: `GroupShuffleSplit` by `game_play` (80/20 train/val)
**Rationale**: Prevent data leakage from same-play observations appearing in both sets.

---

## Results

| Metric               | Value   |
|----------------------|---------|
| **Validation ROC-AUC** | **0.9871** |
| Best Iteration       | 117     |
| Training Samples     | 4.7M    |
| Game Plays (train)   | 240     |

### Hyperparameters

```python
{
  "objective": "binary",
  "metric": "auc",
  "learning_rate": 0.05,
  "num_leaves": 64,
  "feature_fraction": 0.9,
  "bagging_fraction": 0.9,
  "bagging_freq": 1,
  "min_data_in_leaf": 50,
  "seed": 42
}
```

Early stopping after 100 rounds without validation improvement.

---

## Next Steps

- **Temporal context**: Add lag features (distance/velocity trends over prior N frames)
- **Multi-modal**: Incorporate helmet sensor accelerometer data
- **Video**: Explore frame-level CNN embeddings for helmet box proximity
- **Production**: Wrap inference in CLI for real-time prediction on new game files

---

## Data

This project uses the [NFL Player Contact Detection](https://www.kaggle.com/competitions/nfl-player-contact-detection) dataset from Kaggle. Raw CSVs are stored locally in `data/` and excluded from version control.
