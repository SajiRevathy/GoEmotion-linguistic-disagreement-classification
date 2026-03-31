import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score,
)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURES_FILE

# ── 1. Load ────────────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_FILE)

# ── 2. Define features ─────────────────────────────────────────────────────
# Excluded: disagreement_score, annotator_all_labels, num_annotators
# These leak the target label directly

FEATURE_COLS = [
    "sent_length",
    "negation_count",
    "hedge_count",
    "pronoun_count",
    "uncertainty_count",
    "contrast_count",
    "sarcasm_markers",
    "intensity_count",
    "exclamation_count",
    "verb_count",
    "dep_depth",
    "clause_count",
    "modal_count",
    "type_token_ratio",
    "is_question",
    "emotion_diversity",
    "emotion_intensity",
    "emotional_conflict",
    "polarity",
]

# Safety check — only keep columns that exist in the file
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

# Warn if any expected features are missing
all_expected = [
    "sent_length", "negation_count", "hedge_count", "pronoun_count",
    "uncertainty_count", "contrast_count", "intensity_count",
    "exclamation_count", "sarcasm_markers", "dep_depth", "clause_count",
    "modal_count", "type_token_ratio", "is_question", "polarity",
    "verb_count", "emotion_diversity", "emotion_intensity",
    "emotional_conflict",
]
missing = [c for c in all_expected if c not in df.columns]
if missing:
    print(f"WARNING — missing features: {missing}")
    print("Re-run feature_extraction.py to regenerate FEATURES_FILE")

TARGET_COL = "disagreement_category"

df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ── 3. Class balance ───────────────────────────────────────────────────────
print("Class distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True).round(3))

# ── 4. Train/test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ── 5. Scale ───────────────────────────────────────────────────────────────
# NOTE: Random Forest does not technically need scaling
# because it uses decision trees which are scale-invariant
# We scale anyway to keep the pipeline consistent with
# logistic regression for fair comparison
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 6. Train ───────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,          # number of trees — more is better up to a point
    class_weight="balanced",   # handles high/low imbalance
    max_depth=None,            # trees grow fully — captures non-linear patterns
    min_samples_leaf=5,        # prevents overfitting on tiny leaf nodes
    max_features="sqrt",       # each tree sees sqrt(19) ≈ 4 features — standard
    random_state=42,
    n_jobs=-1,                 # use all CPU cores — speeds up training
)
model.fit(X_train_sc, y_train)

# ── 7. Evaluate ────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

bal_acc = balanced_accuracy_score(y_test, y_pred)
mcc     = matthews_corrcoef(y_test, y_pred)
print(f"Balanced accuracy:         {bal_acc:.3f}")
print(f"Matthews corr coefficient: {mcc:.3f}")

cv_scores = cross_val_score(
    model, X_train_sc, y_train, cv=5, scoring="f1_macro"
)
print(f"\n5-fold CV F1 macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Baseline comparison
majority_class = y_train.value_counts().idxmax()
y_dummy        = [majority_class] * len(y_test)
dummy_f1       = f1_score(y_test, y_dummy, average="macro")
model_f1       = f1_score(y_test, y_pred, average="macro")
print(f"\nBaseline (always predict '{majority_class}'): {dummy_f1:.3f}")
print(f"Your model F1 macro:                         {model_f1:.3f}")
print(f"Improvement over baseline:                  +{model_f1 - dummy_f1:.3f}")

# ── 8. Confusion matrix ────────────────────────────────────────────────────
cm   = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion matrix — Random Forest disagreement classifier")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png", dpi=150)
plt.show()

# ── 9. Feature importance ──────────────────────────────────────────────────
# Random Forest uses feature importance instead of coefficients
# Importance = how much each feature reduces impurity across all trees
# Range: 0 to 1, all importances sum to 1
# Unlike LR coefficients, these are always positive
# Higher = more important for the decision regardless of direction
importance_df = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nFeature importances (higher = more useful for prediction):")
print(importance_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(
    importance_df["feature"],
    importance_df["importance"],
    color="#534AB7"
)
ax.set_xlabel("Importance (higher = more useful)")
ax.set_title("Random Forest — feature importance")
plt.tight_layout()
plt.savefig("feature_importance_rf.png", dpi=150)
plt.show()

# ── 10. Comparison table ───────────────────────────────────────────────────
# Print a clean comparison for your paper
print("\n── Model comparison ──────────────────────────────────────────")
print(f"{'Model':<25} {'F1 macro':<12} {'Bal. Acc':<12} {'MCC':<8}")
print(f"{'─'*55}")
print(f"{'Majority baseline':<25} {'0.432':<12} {'0.500':<12} {'0.000':<8}")
print(f"{'Logistic Regression':<25} {'0.528':<12} {'0.575':<12} {'0.128':<8}")
print(f"{'Random Forest':<25} {model_f1:<12.3f} {bal_acc:<12.3f} {mcc:<8.3f}")
print(f"{'─'*55}")
