# EDA script to summarise the final disagreement-labelled dataset.
# Checks the distribution of disagreement categories and their percentages,
# and prints score statistics to verify the disagreement scoring worked correctly.

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DISAGREEMENT_FILE

df = pd.read_csv(DISAGREEMENT_FILE)

print("FINAL SUMMARY")
print(f"Total comments : {len(df):,}")
print("\nDisagreement categories:")
print(df['disagreement_category'].value_counts().to_string())
print(df['disagreement_category'].value_counts(
      normalize=True).mul(100).round(1).astype(str).add('%').to_string())

print("\nScore statistics:")
print(df['disagreement_score'].describe().round(4).to_string())