# Linguistic Analysis of Annotator Disagreement in Emotion Data

## Pipeline Overview

```
Raw GoEmotions data
    ↓
dataset_download.py           →  downloads and combines raw GoEmotions CSVs
    ↓
annotator_level_cleaning.py   →  cleans and filters annotator-level data
    ↓
group_annotators_by_comment.py →  groups annotations per comment
    ↓
disagreement_score.py         →  computes annotator disagreement via normalized entropy
    ↓
feature_extraction.py         →  extracts 22 linguistic features per comment
    ↓
train_logistic_regression.py  →  trains logistic regression classifier, outputs feature importance
train_random_forest.py        →  trains random forest classifier
```

---

## Folder Structure

```
linguistic-disagreement-classification/
│
├── data/
│   ├── raw/                              ← original downloaded GoEmotions CSVs
│   └── processed/                        ← cleaned and feature-extracted outputs
│
├── src/
│   ├── eda/                              ← exploratory data analysis scripts
│   ├── plots/                            ← generated visualisations
│   ├── annotator_level_cleaning.py       ← cleans raw annotator data
│   ├── group_annotators_by_comment.py    ← groups rater annotations per comment
│   ├── disagreement_score.py             ← computes normalized entropy scores
│   ├── feature_extraction.py             ← extracts 22 linguistic features
│   ├── train_logistic_regression.py      ← trains LR model, ranks features
│   └── train_random_forest.py            ← trains Random Forest model
│
├── config.py                             ← all file paths in one place
├── dataset_download.py                   ← downloads raw GoEmotions dataset
└── README.md
```

---

## Dataset

**GoEmotions** — Demszky et al. (2020), Google Research
- 58,000 English Reddit comments across 27 emotion categories + Neutral (total = 28)
- Each comment rated by multiple annotators
- Annotator-level ratings used to compute disagreement scores via normalized entropy
- Split across 3 raw CSV files (`goemotions_1.csv`, `goemotions_2.csv`, `goemotions_3.csv`)

Download: [Google Research — GoEmotions](https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/)

---

## Features Extracted (22)

| Feature | Description |
|---|---|
| `sent_length` | Number of tokens in the comment |
| `negation_count` | Count of negation words (not, never, no...) |
| `hedge_count` | Count of hedging words (maybe, might, possibly...) |
| `pronoun_count` | Count of pronouns — group targeting signal |
| `verb_count` | Count of verbs — action intensity |
| `uncertainty_count` | Count of uncertainty words (unclear, ambiguous...) |
| `contrast_count` | Count of contrast words (but, however, although...) |
| `sarcasm_markers` | Punctuation/typographic sarcasm signals |
| `intensity_count` | Count of intensifiers (absolutely, extremely...) |
| `exclamation_count` | Number of exclamation marks |
| `dep_depth` | Syntactic depth — grammatical complexity |
| `clause_count` | Number of subordinate clauses |
| `modal_count` | Count of modal verbs (should, would, must...) |
| `type_token_ratio` | Lexical diversity (0=repetitive, 1=all unique) |
| `is_question` | Whether the comment ends with a question mark |
| `polarity` | Overall sentiment (-1=negative, +1=positive) |
| `emotion_diversity` | Whether both positive and negative tones are present |
| `emotion_intensity` | Strength of overall emotion (0=weak, 1=strong) |
| `emotional_conflict` | Whether comment pulls in both emotional directions |
| `profanity_count` | Count of profane/offensive words |
| `allcaps_ratio` | Proportion of words written in ALL CAPS |
| `avg_token_length` | Average word length — proxy for language complexity |

---

## Tools Used

| Tool | Purpose |
|---|---|
| spaCy `en_core_web_sm` | POS tagging, dependency parsing, syntactic features |
| TextBlob | Sentiment polarity |
| VADER (NLTK) | Emotion intensity and emotional conflict detection |
| better_profanity | Profanity detection |
| Custom word lists | Hedging, negation, contrast, uncertainty, intensity |
| Scikit-learn | Logistic Regression and Random Forest classifiers |

---

## Setup

```bash
git clone https://github.com/your-username/GoEmotion-linguistic-disagreement-classification.git
cd linguistic-disagreement-classification
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Step 1: Download raw data
python dataset_download.py

# Step 2: Clean and prepare
python src/annotator_level_cleaning.py
python src/group_annotators_by_comment.py

# Step 3: Compute disagreement scores
python src/disagreement_score.py

# Step 4: Extract features
python src/feature_extraction.py

# Step 5: Train models
python src/train_logistic_regression.py
python src/train_random_forest.py
```
