# Cleans the raw GoEmotions dataset at the annotator level.
# Removes rows where annotator chose no emotion (emotion_count=0),
# extracts the primary emotion for multi-label rows,
# keeps only useful columns and saves the cleaned result to processed folder.

import pandas as pd
import sys
from pathlib import Path


# ── Load ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import RAW_FILE, PROC_DIR, ANNOTATOR_CLEAN_FILE
df       = pd.read_csv(RAW_FILE)
print(f"Loaded! Total rows: {len(df):,}")

# ── Your exact emotion columns ────────────────────────────────────────
EMOTION_COLS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness',
    'surprise', 'neutral'
]

# ── Recalculate emotion count ─────────────────────────────────────────
df['emotion_count'] = df[EMOTION_COLS].sum(axis=1)

# ─────────────────────────────────────────────────────────────────────
# BEFORE CLEANING SNAPSHOT
# ─────────────────────────────────────────────────────────────────────
print("\nBEFORE CLEANING:")
print(f"   Total rows          : {len(df):,}")
print(f"   0 emotions (remove) : {(df['emotion_count']==0).sum():,}")
print(f"   1 emotion  (keep)   : {(df['emotion_count']==1).sum():,}")
print(f"   2+ emotions (handle): {(df['emotion_count']>=2).sum():,}")

# ─────────────────────────────────────────────────────────────────────
# STEP 1 — Remove rows where annotator chose nothing (emotion_count=0)
# ─────────────────────────────────────────────────────────────────────
df_clean = df[df['emotion_count'] >= 1].copy()
removed  = len(df) - len(df_clean)
print(f"\nStep 1 — Removed {removed:,} rows where annotator chose nothing")
print(f"   Remaining: {len(df_clean):,} rows")

# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Handle multiple emotions
# Take the FIRST emotion in ordered list = most specific emotion
# ─────────────────────────────────────────────────────────────────────
def get_primary_emotion(row):
    chosen = [col for col in EMOTION_COLS if row[col] == 1]
    return chosen[0]  # first = most specific

df_clean['primary_emotion'] = df_clean.apply(get_primary_emotion, axis=1)
print("\nStep 2 — Primary emotion extracted for all rows")

# ─────────────────────────────────────────────────────────────────────
# KEEP ONLY YOUR USEFUL COLUMNS
# Using your exact column names from the dataset
# ─────────────────────────────────────────────────────────────────────
keep_cols = [
    'id',              # comment ID      → to group annotators
    'text',            # comment text    → for linguistic features
    'rater_id',        # annotator ID    → who labelled it
    'emotion_count',   # how many emotions chosen
    'primary_emotion'  # final clean label
]

df_clean = df_clean[keep_cols]

# ─────────────────────────────────────────────────────────────────────
# AFTER CLEANING SNAPSHOT
# ─────────────────────────────────────────────────────────────────────
print("\nAFTER CLEANING:")
print(f"   Total rows remaining    : {len(df_clean):,}")
print(f"   Columns kept            : {keep_cols}")
print("\n   Primary emotion distribution:")
print(df_clean['primary_emotion'].value_counts().head(10).to_string())

print("\nSAMPLE — cleaned data:")
print(df_clean.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────
PROC_DIR.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(ANNOTATOR_CLEAN_FILE, index=False)

print("\nSaved to: {ANNOTATOR_CLEAN_FILE}")
print("\nFINAL SUMMARY:")
print(f"   Original rows    : {len(df):,}")
print(f"   Removed rows     : {removed:,}")
print(f"   Final rows       : {len(df_clean):,}")
print(f"   Columns          : {keep_cols}")
