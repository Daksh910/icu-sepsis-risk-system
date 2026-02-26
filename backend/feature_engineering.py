import pandas as pd
import numpy as np

def preprocess_data(filepath):

    data = pd.read_csv(filepath)

    # Remove duplicates
    data = data.drop_duplicates()

    # Fill missing values
    data = data.ffill()
    data = data.fillna(data.median(numeric_only=True))
    data = data.fillna(data.median())

    # Group by patient (assuming patient ID exists)
    if "PatientID" in data.columns:
        grouped = data.groupby("PatientID")
    else:
        # If no PatientID, assume each file is per patient
        grouped = data.groupby(data.index)

    feature_list = []

    for patient_id, group in grouped:

        features = {}

        # Aggregate vitals
        features["HR_mean"] = group["HR"].mean()
        features["HR_max"] = group["HR"].max()
        features["Temp_max"] = group["Temp"].max()
        features["MAP_min"] = group["MAP"].min()
        features["Resp_mean"] = group["Resp"].mean()
        features["O2Sat_min"] = group["O2Sat"].min()

        # Lab summaries
        features["Lactate_mean"] = group["Lactate"].mean()
        features["WBC_mean"] = group["WBC"].mean()

        # Demographics
        features["Age"] = group["Age"].iloc[0]
        features["Gender"] = group["Gender"].iloc[0]

        # Target
        features["Sepsis"] = group["SepsisLabel"].max()

        feature_list.append(features)

    final_df = pd.DataFrame(feature_list)

    return final_df