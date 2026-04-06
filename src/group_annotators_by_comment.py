# Groups the cleaned annotator-level data by comment id.
# Each comment gets one row with all annotator emotions stored as a list,
# filters to comments with 3-5 annotators only,
# creates individual ann1-ann5 emotion columns for each annotator,
# and saves the grouped result for downstream disagreement analysis.

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ANNOTATOR_CLEAN_FILE, GROUPED_FILE, PROC_DIR

# ── Load cleaned annotator file ───────────────────────────────────────
df = pd.read_csv(ANNOTATOR_CLEAN_FILE)
print(f"Loaded! Total rows: {len(df):,}")

# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Group by comment id
# Each comment gets ONE row with all annotator labels
# ─────────────────────────────────────────────────────────────────────
def aggregate_comment(group):
    labels = group['primary_emotion'].tolist()
    return pd.Series({
        'text'            : group['text'].iloc[0],
        'all_labels'      : labels,
        'annotator_count' : len(labels)
    })

grouped = df.groupby('id', group_keys=False).apply(aggregate_comment,include_groups=False).reset_index()
print(f"Grouped! Unique comments: {len(grouped):,}")

# ─────────────────────────────────────────────────────────────────────
# CHECK annotator count distribution
# ─────────────────────────────────────────────────────────────────────
print("\nANNOTATOR COUNT DISTRIBUTION:")
print(grouped['annotator_count'].value_counts().sort_index().to_string())
print(f"\nMin : {grouped['annotator_count'].min()}")
print(f"Max : {grouped['annotator_count'].max()}")
print(f"Mean: {grouped['annotator_count'].mean():.2f}")

# ── Filter to 3-5 annotators ──────────────────────────────────────────
before  = len(grouped)
grouped = grouped[grouped['annotator_count'].between(3, 5)].copy()
after   = len(grouped)

print("\nFiltered to 3-5 annotators")
print(f"   Before  : {before:,}")
print(f"   After   : {after:,}")
print(f"   Removed : {before - after:,}")

# ── Create individual annotator columns ───────────────────────────────
max_ann = grouped['annotator_count'].max()  # will be 5
for i in range(max_ann):
    col = f'ann{i+1}_emotion'
    grouped[col] = grouped['all_labels'].apply(
        lambda x, i=i: x[i] if i < len(x) else None
    )
print(f"Created columns: ann1_emotion to ann{max_ann}_emotion")

# ── Final columns ─────────────────────────────────────────────────────
ann_cols   = [f'ann{i+1}_emotion' for i in range(max_ann)]
final_cols = ['id', 'text'] + ann_cols + ['annotator_count', 'all_labels']
grouped    = grouped[final_cols]

# ── Sample ────────────────────────────────────────────────────────────
print("\nSAMPLE:")
print(grouped[['id', 'text', 'ann1_emotion', 'ann2_emotion',
               'ann3_emotion', 'ann4_emotion', 'ann5_emotion',
               'annotator_count']].head(5).to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────
PROC_DIR.mkdir(parents=True, exist_ok=True)
grouped.to_csv(GROUPED_FILE, index=False)

# ── Final summary ─────────────────────────────────────────────────────
print("\n FINAL SUMMARY:")
print(f"   Total comments : {len(grouped):,}")
print("\n   Annotator breakdown:")
print(grouped['annotator_count'].value_counts().sort_index().to_string())
print(f"\n Saved to: {GROUPED_FILE}")
