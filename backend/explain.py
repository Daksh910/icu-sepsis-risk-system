import shap
import joblib
import pandas as pd
from feature_engineering import preprocess_data
import matplotlib.pyplot as plt

print("Loading model...")
model = joblib.load("saved_models/triage_model.pkl")

print("Loading dataset...")
data = preprocess_data("data/sepsis.csv")
X = data.drop("Sepsis", axis=1)

print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X)