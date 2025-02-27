import os
from fastapi import FastAPI
import numpy as np
import pandas as pd
from scripts.utils import load_pipeline, create_pickle_file
from pydantic import BaseModel, validator
from typing import Optional

app = FastAPI(title="ICU Mortality Prediction API", description="API to predict ICU mortality", version="1.0")

model = load_pipeline("artifacts/model.pkl")
imputer = load_pipeline("artifacts/imputer.pkl")
scaler = None
pca = None
try:
    scaler = load_pipeline("artifacts/scaler.pkl")
    pca = load_pipeline("artifacts/pca.pkl")
except:
    pass

class PatientData(BaseModel):
    hospital_id: Optional[int] = np.nan
    age: Optional[float] = np.nan
    bmi: Optional[float] = np.nan
    elective_surgery: Optional[int] = np.nan
    ethnicity: Optional[str] = np.nan
    gender: Optional[str] = np.nan
    height: Optional[float] = np.nan
    icu_admit_source: Optional[str] = np.nan
    icu_id: Optional[int] = np.nan
    icu_stay_type: Optional[str] = np.nan
    icu_type: Optional[str] = np.nan
    pre_icu_los_days: Optional[float] = np.nan
    weight: Optional[float] = np.nan
    apache_2_diagnosis: Optional[int] = np.nan
    apache_3j_diagnosis: Optional[float] = np.nan
    apache_post_operative: Optional[int] = np.nan
    arf_apache: Optional[int] = np.nan
    gcs_eyes_apache: Optional[int] = np.nan
    gcs_motor_apache: Optional[int] = np.nan
    gcs_unable_apache: Optional[int] = np.nan
    gcs_verbal_apache: Optional[int] = np.nan
    heart_rate_apache: Optional[int] = np.nan
    intubated_apache: Optional[int] = np.nan
    map_apache: Optional[float] = np.nan
    resprate_apache: Optional[float] = np.nan
    temp_apache: Optional[float] = np.nan
    ventilated_apache: Optional[int] = np.nan
    d1_diasbp_max: Optional[float] = np.nan
    d1_diasbp_min: Optional[float] = np.nan
    d1_diasbp_noninvasive_max: Optional[float] = np.nan
    d1_diasbp_noninvasive_min: Optional[float] = np.nan
    d1_heartrate_max: Optional[float] = np.nan
    d1_heartrate_min: Optional[float] = np.nan
    d1_mbp_max: Optional[float] = np.nan
    d1_mbp_min: Optional[float] = np.nan
    d1_mbp_noninvasive_max: Optional[float] = np.nan
    d1_mbp_noninvasive_min: Optional[float] = np.nan
    d1_resprate_max: Optional[float] = np.nan
    d1_resprate_min: Optional[float] = np.nan
    d1_spo2_max: Optional[float] = np.nan
    d1_spo2_min: Optional[float] = np.nan
    d1_sysbp_max: Optional[float] = np.nan
    d1_sysbp_min: Optional[float] = np.nan
    d1_sysbp_noninvasive_max: Optional[float] = np.nan
    d1_sysbp_noninvasive_min: Optional[float] = np.nan
    d1_temp_max: Optional[float] = np.nan
    d1_temp_min: Optional[float] = np.nan
    h1_diasbp_max: Optional[float] = np.nan
    h1_diasbp_min: Optional[float] = np.nan
    h1_diasbp_noninvasive_max: Optional[float] = np.nan
    h1_diasbp_noninvasive_min: Optional[float] = np.nan
    h1_heartrate_max: Optional[float] = np.nan
    h1_heartrate_min: Optional[float] = np.nan
    h1_mbp_max: Optional[float] = np.nan
    h1_mbp_min: Optional[float] = np.nan
    h1_mbp_noninvasive_max: Optional[float] = np.nan
    h1_mbp_noninvasive_min: Optional[float] = np.nan
    h1_resprate_max: Optional[float] = np.nan
    h1_resprate_min: Optional[float] = np.nan
    h1_spo2_max: Optional[float] = np.nan
    h1_spo2_min: Optional[float] = np.nan
    h1_sysbp_max: Optional[float] = np.nan
    h1_sysbp_min: Optional[float] = np.nan
    h1_sysbp_noninvasive_max: Optional[float] = np.nan
    h1_sysbp_noninvasive_min: Optional[float] = np.nan
    d1_glucose_max: Optional[float] = np.nan
    d1_glucose_min: Optional[float] = np.nan
    d1_potassium_max: Optional[float] = np.nan
    d1_potassium_min: Optional[float] = np.nan
    apache_4a_hospital_death_prob: Optional[float] = np.nan
    apache_4a_icu_death_prob: Optional[float] = np.nan
    aids: Optional[int] = np.nan
    cirrhosis: Optional[int] = np.nan
    diabetes_mellitus: Optional[int] = np.nan
    hepatic_failure: Optional[int] = np.nan
    immunosuppression: Optional[int] = np.nan
    leukemia: Optional[int] = np.nan
    lymphoma: Optional[int] = np.nan
    solid_tumor_with_metastasis: Optional[int] = np.nan
    apache_3j_bodysystem: Optional[str] = np.nan
    apache_2_bodysystem: Optional[str] = np.nan

class FeedbackData(BaseModel):
    hospital_id: Optional[int] = np.nan
    age: Optional[float] = np.nan
    bmi: Optional[float] = np.nan
    elective_surgery: Optional[int] = np.nan
    ethnicity: Optional[str] = np.nan
    gender: Optional[str] = np.nan
    height: Optional[float] = np.nan
    icu_admit_source: Optional[str] = np.nan
    icu_id: Optional[int] = np.nan
    icu_stay_type: Optional[str] = np.nan
    icu_type: Optional[str] = np.nan
    pre_icu_los_days: Optional[float] = np.nan
    weight: Optional[float] = np.nan
    apache_2_diagnosis: Optional[int] = np.nan
    apache_3j_diagnosis: Optional[float] = np.nan
    apache_post_operative: Optional[int] = np.nan
    arf_apache: Optional[int] = np.nan
    gcs_eyes_apache: Optional[int] = np.nan
    gcs_motor_apache: Optional[int] = np.nan
    gcs_unable_apache: Optional[int] = np.nan
    gcs_verbal_apache: Optional[int] = np.nan
    heart_rate_apache: Optional[int] = np.nan
    intubated_apache: Optional[int] = np.nan
    map_apache: Optional[float] = np.nan
    resprate_apache: Optional[float] = np.nan
    temp_apache: Optional[float] = np.nan
    ventilated_apache: Optional[int] = np.nan
    d1_diasbp_max: Optional[float] = np.nan
    d1_diasbp_min: Optional[float] = np.nan
    d1_diasbp_noninvasive_max: Optional[float] = np.nan
    d1_diasbp_noninvasive_min: Optional[float] = np.nan
    d1_heartrate_max: Optional[float] = np.nan
    d1_heartrate_min: Optional[float] = np.nan
    d1_mbp_max: Optional[float] = np.nan
    d1_mbp_min: Optional[float] = np.nan
    d1_mbp_noninvasive_max: Optional[float] = np.nan
    d1_mbp_noninvasive_min: Optional[float] = np.nan
    d1_resprate_max: Optional[float] = np.nan
    d1_resprate_min: Optional[float] = np.nan
    d1_spo2_max: Optional[float] = np.nan
    d1_spo2_min: Optional[float] = np.nan
    d1_sysbp_max: Optional[float] = np.nan
    d1_sysbp_min: Optional[float] = np.nan
    d1_sysbp_noninvasive_max: Optional[float] = np.nan
    d1_sysbp_noninvasive_min: Optional[float] = np.nan
    d1_temp_max: Optional[float] = np.nan
    d1_temp_min: Optional[float] = np.nan
    h1_diasbp_max: Optional[float] = np.nan
    h1_diasbp_min: Optional[float] = np.nan
    h1_diasbp_noninvasive_max: Optional[float] = np.nan
    h1_diasbp_noninvasive_min: Optional[float] = np.nan
    h1_heartrate_max: Optional[float] = np.nan
    h1_heartrate_min: Optional[float] = np.nan
    h1_mbp_max: Optional[float] = np.nan
    h1_mbp_min: Optional[float] = np.nan
    h1_mbp_noninvasive_max: Optional[float] = np.nan
    h1_mbp_noninvasive_min: Optional[float] = np.nan
    h1_resprate_max: Optional[float] = np.nan
    h1_resprate_min: Optional[float] = np.nan
    h1_spo2_max: Optional[float] = np.nan
    h1_spo2_min: Optional[float] = np.nan
    h1_sysbp_max: Optional[float] = np.nan
    h1_sysbp_min: Optional[float] = np.nan
    h1_sysbp_noninvasive_max: Optional[float] = np.nan
    h1_sysbp_noninvasive_min: Optional[float] = np.nan
    d1_glucose_max: Optional[float] = np.nan
    d1_glucose_min: Optional[float] = np.nan
    d1_potassium_max: Optional[float] = np.nan
    d1_potassium_min: Optional[float] = np.nan
    apache_4a_hospital_death_prob: Optional[float] = np.nan
    apache_4a_icu_death_prob: Optional[float] = np.nan
    aids: Optional[int] = np.nan
    cirrhosis: Optional[int] = np.nan
    diabetes_mellitus: Optional[int] = np.nan
    hepatic_failure: Optional[int] = np.nan
    immunosuppression: Optional[int] = np.nan
    leukemia: Optional[int] = np.nan
    lymphoma: Optional[int] = np.nan
    solid_tumor_with_metastasis: Optional[int] = np.nan
    apache_3j_bodysystem: Optional[str] = np.nan
    apache_2_bodysystem: Optional[str] = np.nan
    target: Optional[int] = np.nan
    prediction: Optional[int] = np.nan

@app.post("/predict")
async def predict(data: list[PatientData]):
    predictions = []
    for patient in data:
        # Replace None values with np.nan
        data_dict = {k: (v if v is not None else np.nan) for k, v in patient.dict().items()}

        # Convert the data to a numpy array
        data_array = np.array([list(data_dict.values())], dtype=object)
        print(data_array)
        # First, impute missing values
        print("Imputing data...")
        data_transform = imputer.transform(data_array)
        # Then, scale the data and apply PCA if necessary
        if scaler is not None:
            print("Scaling data...")
            data_transform = scaler.transform(data_transform)
            if pca is not None:
                print("Applying PCA...")
                data_transform = pca.transform(data_transform)
        # Finally, make the prediction
        try:
            print("Making prediction...")
            prediction = model.predict(data_transform)
            print("Prediction:", prediction)
            predictions.append(prediction.item())
        except Exception as e:
            return {"error": str(e)}
    return {"predictions": predictions}

@app.post("/feedback")
async def feedback(data: list[FeedbackData]):
    # Create a CSV file with the feedback data

    # Initialize an empty numpy array with the correct shape
    np_data = np.empty((0, len(data[0].dict().keys())), dtype=object)

    for feedback in data:
        # Replace None values with np.nan
        data_dict = {k: (v if v is not None else np.nan) for k, v in feedback.dict().items()}

        # Convert the data to a numpy array
        data_array = np.array([list(data_dict.values())], dtype=object)

        # Append the data array as a new row
        np_data = np.vstack([np_data, data_array])

    # Split the data into features and target
    np_data_features = np_data[:, :-2]
    np_data_target = np_data[:, -2:]

    # Transform the features using the imputer, scaler, and PCA
    np_data_features_transformed = imputer.transform(np_data_features)
    if scaler is not None:
        np_data_features_transformed = scaler.transform(np_data_features_transformed)
        if pca is not None:
            np_data_features_transformed = pca.transform(np_data_features_transformed)

    # Concatenate the transformed features and the target
    np_data = np.hstack((np_data_features_transformed, np_data_target))

    # If prod_data.csv does not exist, create it with the same header as ref_data.csv
    if not os.path.exists("data/prod_data.csv"):
        # Get header from ref_data.csv
        with open("data/ref_data.csv", "r") as f:
            header = f.readline().strip()
            # Append new column to header
            header += ";prediction"
        # Create prod_data.csv with header
        with open("data/prod_data.csv", "w") as f:
            f.write(header + "\n")

    # Append the feedback data at the end of the CSV file prod_data.csv
    with open("data/prod_data.csv", "a") as f:
        np.savetxt(f, np_data, delimiter=";", fmt="%s")

    # Train a new model every 20 feedbacks

    # Get a numpy array from the CSV file
    prod_data = pd.read_csv("data/prod_data.csv", header=1, delimiter=";")

    # Check if the number of feedbacks is a multiple of 20
    if len(prod_data) % 20 == 0:
        # Train a new model
        print("Training a new model...")

        # Split the data into features and target
        np_prod_features = prod_data.iloc[:, :-2]
        np_prod_target = prod_data.iloc[:, -2:-1]
        np_prod_prediction = prod_data.iloc[:, -1:]

        # Load the reference data
        ref_data = pd.read_csv("data/ref_data.csv", header=1, delimiter=";")

        # Split the reference data into features and target
        np_ref_data_features = ref_data.iloc[:, :-1]
        np_ref_data_target = ref_data.iloc[:, -1:]

        # Concatenate the reference data and the feedback data
        np_data = np.vstack([np_ref_data_features, np_prod_features])
        np_target = np.vstack([np_ref_data_target, np_prod_target])

        # Train a new model
        model.fit(np_data, np_target)

        # Save the new model
        create_pickle_file(model, "artifacts/model.pkl")        
