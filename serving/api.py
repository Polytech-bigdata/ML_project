from fastapi import FastAPI
import numpy as np
from scripts.utils import load_pipeline
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
    elective_surgery: Optional[bool] = np.nan
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
    apache_3j_diagnosis: Optional[int] = np.nan
    apache_post_operative: Optional[bool] = np.nan
    arf_apache: Optional[bool] = np.nan
    gcs_eyes_apache: Optional[int] = np.nan
    gcs_motor_apache: Optional[int] = np.nan
    gcs_unable_apache: Optional[bool] = np.nan
    gcs_verbal_apache: Optional[int] = np.nan
    heart_rate_apache: Optional[int] = np.nan
    intubated_apache: Optional[bool] = np.nan
    map_apache: Optional[float] = np.nan
    resprate_apache: Optional[float] = np.nan
    temp_apache: Optional[float] = np.nan
    ventilated_apache: Optional[bool] = np.nan
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
    aids: Optional[bool] = np.nan
    cirrhosis: Optional[bool] = np.nan
    diabetes_mellitus: Optional[bool] = np.nan
    hepatic_failure: Optional[bool] = np.nan
    immunosuppression: Optional[bool] = np.nan
    leukemia: Optional[bool] = np.nan
    lymphoma: Optional[bool] = np.nan
    solid_tumor_with_metastasis: Optional[bool] = np.nan
    apache_3j_bodysystem: Optional[str] = np.nan
    apache_2_bodysystem: Optional[str] = np.nan

@app.post("/predict")
async def predict(data: PatientData):
    # Replace None values with np.nan
    data_dict = {k: (v if v is not None else np.nan) for k, v in data.dict().items()}

    #convert the data to a numpy array
    data_array = np.array([list(data_dict.values())])
    print(data_array)
    #first, impute missing values
    print("Imputing data...")
    data_transform = imputer.transform(data_array)
    #then, scale the data and apply PCA if necessary
    if scaler is not None:
        print("Scaling data...")
        data_transform = scaler.transform(data_transform)
        if pca is not None:
            print("Applying PCA...")
            data_transform = pca.transform(data_transform)
    #finally, make the prediction
    try:
        print("Making prediction...")
        prediction = model.predict(data_transform)
        print("Prediction:", prediction)
    except Exception as e:
        return {"error": str(e)}
    return {"predictions": prediction.tolist()}