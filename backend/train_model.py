import pandas as pd
import numpy as np
import joblib
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, fbeta_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from feature_engineering import preprocess_data

print("\n🔄 Loading and preprocessing data...")
data = preprocess_data("data/sepsis.csv")

X = data.drop("Sepsis", axis=1)
y = data["Sepsis"]

print("✂ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print(f"\n⚖ Imbalance Ratio: {scale_pos_weight:.2f}")

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [500, 700],
    "max_depth": [6, 8],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8, 1]
}

print("\n🔬 Grid Search (PR-AUC)...")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    xgb_model,
    param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

print("\n🏆 Best Params:", grid.best_params_)
print("📈 Best CV PR-AUC:", grid.best_score_)

best_model.fit(X_train_scaled, y_train)

proba = best_model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 Test ROC-AUC:", roc_auc_score(y_test, proba))
print("📊 Test PR-AUC:", average_precision_score(y_test, proba))

# 🔥 Optimize F2 (recall weighted)
thresholds = np.arange(0.1, 0.6, 0.02)
best_score = 0
best_threshold = 0.5

for t in thresholds:
    preds = (proba > t).astype(int)
    score = fbeta_score(y_test, preds, beta=2)
    if score > best_score:
        best_score = score
        best_threshold = t

print(f"\n🎯 Optimal Threshold (F2): {best_threshold}")
print(f"🔥 Best F2 Score: {best_score}")

final_preds = (proba > best_threshold).astype(int)

print("\n📋 Final Report:\n")
print(classification_report(y_test, final_preds, zero_division=0))

print("\n💾 Saving model...")
joblib.dump(best_model, "saved_models/triage_model.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
joblib.dump(best_threshold, "saved_models/threshold.pkl")

print("✅ Done.")