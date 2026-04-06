# step3_disagreement_score.py
# PURPOSE: Calculate disagreement score per comment
#          - Convert all_labels from string to list
#          - Calculate normalised entropy score
#          - Map to binary categories: HIGH and LOW only
# INPUT  : data/processed/goemotions_grouped.csv
# OUTPUT : data/processed/goemotions_with_disagreement.csv

import pandas as pd
import numpy as np
from scipy.stats import entropy
from pathlib import Path

import matplotlib.pyplot as plt
import ast
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GROUPED_FILE, PROC_DIR, DISAGREEMENT_FILE

# Check 1: does the file exist before even trying to read it?
assert Path(GROUPED_FILE).exists(), \
    f"Input file not found: {GROUPED_FILE}"

# ── Load grouped file ─────────────────────────────────────────────────
df = pd.read_csv(GROUPED_FILE)


# Check 2: did we actually get rows?
assert len(df) > 0, \
    f"File loaded but is empty: {GROUPED_FILE}"
    
# Check 3: do the columns we need actually exist?
required_columns = ['text', 'all_labels']
missing = [c for c in required_columns if c not in df.columns]
assert not missing, \
    f"Missing expected columns: {missing}. Found: {list(df.columns)}"
    
print(f"Loaded! Total rows: {len(df):,}")

# ─────────────────────────────────────────────────────────────────────
# FIX — Convert all_labels from string to real Python list
# When saved to CSV lists become strings
# ast.literal_eval converts them back safely
# ─────────────────────────────────────────────────────────────────────
df['all_labels'] = df['all_labels'].apply(ast.literal_eval)

print(f"\n all_labels type after fix: {type(df['all_labels'].iloc[0])}")
print(f"   Example: {df['all_labels'].iloc[0]}",flush=True)

# ─────────────────────────────────────────────────────────────────────
# CALCULATE DISAGREEMENT SCORE
# Uses normalised entropy
# Works fairly for 3, 4, or 5 annotators
# Always returns value between 0.0 and 1.0
#
# 0.0 = full agreement    → all annotators chose same emotion
# 0.5 = partial agreement → some agreed some didn't
# 1.0 = full disagreement → all annotators chose different emotions
# ─────────────────────────────────────────────────────────────────────
def calc_disagreement(labels):
    # Need at least 2 labels
    if len(labels) < 2:
        return np.nan

    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) == 1:
        return 0.0
    probs          = counts / len(labels)
    raw_entropy    = entropy(probs, base=2)
    max_entropy = np.log2(len(unique)) 


    return round(raw_entropy / max_entropy, 4)

df['disagreement_score'] = df['all_labels'].apply(calc_disagreement)
print("\nDisagreement score calculated!")

# ─────────────────────────────────────────────────────────────────────
# MAP TO BINARY CATEGORY — HIGH and LOW only
# score <  threshold → LOW  (below median, annotators mostly agreed)
# score >= threshold → HIGH (at or above median, annotators mostly disagreed)
# ─────────────────────────────────────────────────────────────────────

def categorise(score, thresh):                   # thresh is now visible, threshold passed explicitly as parameter
    if pd.isna(score):     return 'unknown'
    elif score <  thresh:  return 'low'
    else:                  return 'high'
    
# caller supply it — no hidden dependency
threshold = df['disagreement_score'].median()
df['disagreement_category'] = df['disagreement_score'].apply(
    lambda score: categorise(score, threshold)    # explicit — can see exactly what value is used
)

print("Disagreement category assigned (HIGH / LOW only)!")
print(f"Threshold used       : {threshold:.4f}")
print("Method               : median")  # change label to match your choice
# ─────────────────────────────────────────────────────────────────────
# VERIFY — show real examples for each category
# ─────────────────────────────────────────────────────────────────────
print("\nREAL EXAMPLES:")

print("\nLOW disagreement (annotators mostly agreed):")
low = df[df['disagreement_category'] == 'low'][
    ['text', 'all_labels', 'disagreement_score']].head(2)
print(low.to_string(index=False))

print("\nHIGH disagreement (annotators mostly disagreed):")
high = df[df['disagreement_category'] == 'high'][
    ['text', 'all_labels', 'disagreement_score']].head(2)
print(high.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────
PROC_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(DISAGREEMENT_FILE, index=False)



# ─────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL SUMMARY")
print("="*55)
print(f"Total comments          : {len(df):,}")
print("\nDisagreement score stats:")
print(df['disagreement_score'].describe().round(4).to_string())
print("\nDisagreement categories:")
print(df['disagreement_category'].value_counts().to_string())
print(df['disagreement_category'].value_counts(
      normalize=True).mul(100).round(1).astype(str).add('%').to_string())
print(f"\nSaved to: {DISAGREEMENT_FILE}")



plt.figure(figsize=(10, 5))
plt.hist(df['disagreement_score'], bins=50, color='steelblue', edgecolor='white')
plt.axvline(df['disagreement_score'].median(), color='red',    label=f"Median {df['disagreement_score'].median():.3f}")
plt.axvline(df['disagreement_score'].mean(),   color='orange', label=f"Mean {df['disagreement_score'].mean():.3f}")
plt.axvline(0.5, color='green', label='Fixed 0.5')
plt.legend()
plt.title('Disagreement Score Distribution')
plt.xlabel('Disagreement Score')
plt.ylabel('Number of Comments')
plt.savefig('disagreement_distribution.png', dpi=150)
plt.show()