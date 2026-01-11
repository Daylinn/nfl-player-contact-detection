from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = REPO_ROOT  # because your CSVs are currently in repo root
# If you later move CSVs into data/raw/, change to:
# DATA_RAW = REPO_ROOT / "data" / "raw" 