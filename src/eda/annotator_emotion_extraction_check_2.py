# EDA script to verify emotion extraction at the individual annotator level.
# Extracts each annotator's chosen emotion from binary columns into a readable column
# and previews the first 6 rows to confirm the logic works before any grouping or aggregation.


import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import RAW_FILE, PROC_DIR

PROC_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(RAW_FILE)
        
# ── Load ──────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_FILE)
print(f"Loaded! Total rows: {len(df):,}")

# ── Emotion columns ───────────────────────────────────────────────────
EMOTION_COLS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness',
    'surprise', 'neutral'
]

# ── Step 1: Get each annotator's chosen emotion ───────────────────────
def get_chosen_emotion(row):
    chosen = [col for col in EMOTION_COLS if row[col] == 1]
    return chosen[0] if chosen else 'unclear'

df['chosen_emotion'] = df.apply(get_chosen_emotion, axis=1)
print(" Step 1 done — each annotator's emotion extracted")
print(df[['text', 'rater_id', 'chosen_emotion']].head(6))