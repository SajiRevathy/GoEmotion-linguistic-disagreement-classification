# EDA script to analyse how many emotions each annotator chose per comment.
# Breaks down the distribution of 0, 1, 2, 3+ emotion selections,
# and inspects extreme cases (rows with 12 emotions chosen) to understand
# the range and quality of annotations before cleaning.


import pandas as pd
import sys
from pathlib import Path

# ── Load ──────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import RAW_FILE,RAW_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(RAW_FILE)

# ── Emotion columns ───────────────────────────────────────────────────
EMOTION_COLS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness',
    'surprise', 'neutral'
]

print(len(EMOTION_COLS))

# ── Check emotion count per annotator row ─────────────────────────────
df['emotion_count'] = df[EMOTION_COLS].sum(axis=1)

# print("HOW MANY EMOTIONS PER ANNOTATOR ROW:")
# print(df['emotion_count'].value_counts().sort_index())
# print(f"\n0 emotions (chose nothing) : {(df['emotion_count']==0).sum():,}")
# print(f"1 emotion  (normal)        : {(df['emotion_count']==1).sum():,}")
# print(f"2 emotions (multiple)      : {(df['emotion_count']==2).sum():,}")
# print(f"3+ emotions (multiple)     : {(df['emotion_count']>=3).sum():,}")
# print(f"\nMultiple emotions %        : {(df['emotion_count']>=2).mean()*100:.1f}%")
# print(f"Chose nothing %            : {(df['emotion_count']==0).mean()*100:.1f}%")


# Check rows where emotion_count is 5 or more
high_emotion_rows = df[df['emotion_count'] >= 5]

print(f"Rows with 5+ emotions: {len(high_emotion_rows)}")
print("\nLet us look at them:")

# Show only columns where at least one value is 1
# This makes the table much cleaner
selected_cols = ['text', 'rater_id'] + [
    col for col in EMOTION_COLS 
    if high_emotion_rows[col].sum() > 0
]

# Clean table format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 30)

print(high_emotion_rows[selected_cols].head(10).to_string(index=False))
# Check your actual column names
print("YOUR ACTUAL COLUMNS:")
print(df.columns.tolist())

print(f"\nTotal columns: {len(df.columns)}")