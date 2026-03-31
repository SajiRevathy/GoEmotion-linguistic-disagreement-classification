# Quick diagnostic script to verify the dataset loads correctly
# and explore the annotator count distribution per comment

import pandas as pd
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import RAW_FILE


#  Debug: print paths so I can verify before running
print("Looking for file at:")
print(RAW_FILE)
print("File exists:", RAW_FILE.exists())

df = pd.read_csv(RAW_FILE)


# Check annotator count distribution
ann_counts = df.groupby('id')['rater_id'].count()

print("ANNOTATORS PER COMMENT:")
print(ann_counts.value_counts().sort_index())
print(f"\nMin annotators : {ann_counts.min()}")
print(f"Max annotators : {ann_counts.max()}")
print(f"Mean annotators: {ann_counts.mean():.2f}")
