import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
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

# ── 2. Define features — EXCLUDE leaky columns ─────────────────────────────
# disagreement_score directly encodes the label → data leakage
# annotator_all_labels, num_annotators → also leak the label
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

# Only keep columns that actually exist in your file
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

# Warn if any expected features are missing
all_expected = [
    "sent_length", "negation_count", "hedge_count", "uncertainty_count",
    "contrast_count", "intensity_count", "exclamation_count",
    "sarcasm_markers", "dep_depth", "clause_count", "modal_count",
    "type_token_ratio", "is_question", "polarity", "verb_count","pronoun_count",
    "emotion_diversity", "emotion_intensity", "emotional_conflict",
]

missing = [c for c in all_expected if c not in df.columns]
if missing:
    print(f"WARNING — these expected features are missing from your file: {missing}")
    print("Re-run feature_extraction.py first to regenerate FEATURES_FILE")

TARGET_COL = "disagreement_category"   # "low" / "high"

# Drop any rows with missing values in features or label
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ── 3. Check class balance ─────────────────────────────────────────────────
print("Class distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True).round(3))
# GoEmotions is typically very skewed toward 'low' — expect ~80/20 or more

# ── 4. Train/test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # keeps class ratio the same in both splits
)

# ── 5. Scale — CRITICAL for logistic regression ────────────────────────────
# sent_length (0–30+) and subjectivity (0–1) are on very different scales
# without scaling, LR gives too much weight to large-magnitude features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_sc  = scaler.transform(X_test)        # apply same scale to test

# ── 6. Train ───────────────────────────────────────────────────────────────
model = LogisticRegression(
    class_weight="balanced",   # handles low/high imbalance automatically
    max_iter=1000,             # default 100 often fails to converge
    random_state=42,
    C=1.0,                     # regularisation strength (tune this later)
    solver="lbfgs",
)
model.fit(X_train_sc, y_train)

# ── 7. Evaluate ────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Balanced accuracy — accuracy that accounts for imbalance
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced accuracy: {bal_acc:.3f}")

# Matthews Correlation Coefficient — best single metric for imbalance
# Range: -1 (perfectly wrong) to +1 (perfectly right), 0 = random
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews correlation coefficient: {mcc:.3f}")

# Cross-validation (more reliable than single split)
cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="f1_macro")
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
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion matrix — disagreement classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ── 9. Feature importance (coefficients) ──────────────────────────────────
# This directly tests your thesis: which features predict disagreement?
coef_df = pd.DataFrame({
    "feature":     FEATURE_COLS,
    "coefficient": model.coef_[0],
}).sort_values("coefficient", ascending=False)

print("\nFeature coefficients (positive = predicts 'high' disagreement):")
print(coef_df.to_string(index=False))

# Plot it
fig, ax = plt.subplots(figsize=(9, 8))
colors = ["#D85A30" if c > 0 else "#378ADD" for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Coefficient (positive = more disagreement)")
ax.set_title("Logistic regression feature importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()