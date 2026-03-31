# EDA script to explore and choose the best threshold for binary disagreement classification.
# Analyses the disagreement score distribution using median, mean, and percentiles,
# then shows the resulting HIGH/LOW class balance at each threshold
# to help decide the most balanced split before building the classifier.

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DISAGREEMENT_FILE

df = pd.read_csv(DISAGREEMENT_FILE)

# ── Method 1: Median threshold ────────────────────────────────────────
# Split at the middle of your actual score distribution
median_threshold = df['disagreement_score'].median()
mean_threshold   = df['disagreement_score'].mean()

print("YOUR SCORE DISTRIBUTION:")
print(df['disagreement_score'].describe().round(4).to_string())

print("\nDATA-DRIVEN THRESHOLD OPTIONS:")
print(f"   Median (50th percentile) : {median_threshold:.4f}")
print(f"   Mean                     : {mean_threshold:.4f}")
print(f"   25th percentile          : {df['disagreement_score'].quantile(0.25):.4f}")
print(f"   75th percentile          : {df['disagreement_score'].quantile(0.75):.4f}")

# ── Method 2: Try each threshold and see balance ──────────────────────
print("\nBALANCE AT EACH THRESHOLD:")
print(f"{'Threshold':<12} {'HIGH':>8} {'LOW':>8} {'HIGH%':>8} {'LOW%':>8}")
print("-" * 50)

thresholds = [
    df['disagreement_score'].quantile(0.25),
    df['disagreement_score'].quantile(0.50),
    df['disagreement_score'].quantile(0.75),
    mean_threshold,
]

for t in thresholds:
    high  = (df['disagreement_score'] >= t).sum()
    low   = (df['disagreement_score'] <  t).sum()
    total = len(df)
    print(f"{t:<12.4f} {high:>8,} {low:>8,} "
          f"{high/total*100:>7.1f}% {low/total*100:>7.1f}%")