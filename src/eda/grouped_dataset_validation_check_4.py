# EDA script to validate the grouped GoEmotions dataset after grouping annotators.
# Checks for missing values, verifies all_labels column format,
# inspects annotator count distribution, and previews sample rows
# to confirm the grouping step worked correctly before further processing.

import pandas as pd
from pathlib import Path
import sys
from config import GROUPED_FILE

# ── Load grouped file ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
df = pd.read_csv(GROUPED_FILE)

print(f"Loaded! Total rows: {len(df):,}")

# ── Check 1: Any missing texts? ───────────────────────────────────────
print("\nCHECK 1 — Missing values:")
print(df[['text', 'ann1_emotion', 'ann2_emotion', 
          'ann3_emotion', 'all_labels']].isnull().sum().to_string())

# ── Check 2: all_labels column format ────────────────────────────────
print("\nCHECK 2 — all_labels format (first 5 rows):")
print(df['all_labels'].head())
print(f"\nType of first value: {type(df['all_labels'].iloc[0])}")

# ── Check 3: annotator count distribution ────────────────────────────
print("\nCHECK 3 — Annotator count distribution:")
print(df['annotator_count'].value_counts().sort_index().to_string())

# ── Check 4: Sample of all_labels vs ann columns ─────────────────────
print("\nCHECK 4 — Sample comparison:")
print(df[['text', 'all_labels', 'annotator_count']].head(3).to_string())