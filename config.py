# config.py
from pathlib import Path

# Project root — works on any system
PROJECT_ROOT = Path(__file__).resolve().parent

# Directories
RAW_DIR  = PROJECT_ROOT / 'data' / 'raw'
PROC_DIR = PROJECT_ROOT / 'data' / 'processed'
SRC_DIR  = PROJECT_ROOT / 'src'

# Raw files
RAW_FILE = RAW_DIR / 'goemotions_full.csv'

# Processed files
ANNOTATOR_CLEAN_FILE = PROC_DIR / 'goemotions_annotator_clean.csv'
GROUPED_FILE         = PROC_DIR / 'goemotions_grouped.csv'
DISAGREEMENT_FILE    = PROC_DIR / 'goemotions_with_disagreement.csv'
FEATURES_FILE        = PROC_DIR / 'goemotions_with_features.csv'