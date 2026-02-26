import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler

from feature_engineering import preprocess_data

print("🔄 Loading dataset...")
data = preprocess_data("data/sepsis.csv")

X = data.drop("Sepsis", axis=1)
y = data["Sepsis"]

print("✂ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("📦 Loading model artifacts...")
model = joblib.load("saved_models/triage_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
threshold = joblib.load("saved_models/threshold.pkl")

X_test_scaled = scaler.transform(X_test)

print("📊 Generating predictions...")
proba = model.predict_proba(X_test_scaled)[:, 1]
preds = (proba > threshold).astype(int)

# ------------------------
# METRICS
# ------------------------

roc_auc = roc_auc_score(y_test, proba)
pr_auc = average_precision_score(y_test, proba)

print("\n📈 Performance Metrics")
print("----------------------------")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, preds, zero_division=0))

# ------------------------
# CONFUSION MATRIX
# ------------------------

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("saved_models/confusion_matrix.png")
plt.close()

# ------------------------
# ROC CURVE
# ------------------------

fpr, tpr, _ = roc_curve(y_test, proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("saved_models/roc_curve.png")
plt.close()

# ------------------------
# PR CURVE
# ------------------------

precision, recall, _ = precision_recall_curve(y_test, proba)

plt.figure()
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("saved_models/pr_curve.png")
plt.close()

# ------------------------
# CALIBRATION CURVE
# ------------------------

prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curve")
plt.tight_layout()
plt.savefig("saved_models/calibration_curve.png")
plt.close()

print("\n✅ Evaluation complete.")
print("📁 Plots saved inside /saved_models/")