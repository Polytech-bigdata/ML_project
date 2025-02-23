from fastapi import FastAPI
from scripts.utils import load_pipeline
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

model = load_pipeline("artifacts/model.pkl")

class PatientData(BaseModel):
    hospital_id: int
    age: float
    bmi: Optional[float] = None
    elective_surgery: bool
    ethnicity: str
    gender: str
    height: Optional[float] = None
    icu_admit_source: str
    icu_id: int
    icu_stay_type: str
    icu_type: str
    pre_icu_los_days: Optional[float] = None
    weight: Optional[float] = None
    apache_2_diagnosis: Optional[int] = None
    apache_3j_diagnosis: Optional[int] = None
    apache_post_operative: Optional[bool] = None
    arf_apache: Optional[bool] = None
    gcs_eyes_apache: Optional[int] = None
    gcs_motor_apache: Optional[int] = None
    gcs_unable_apache: Optional[bool] = None
    gcs_verbal_apache: Optional[int] = None
    heart_rate_apache: Optional[int] = None
    intubated_apache: Optional[bool] = None
    map_apache: Optional[float] = None
    resprate_apache: Optional[float] = None
    temp_apache: Optional[float] = None
    ventilated_apache: Optional[bool] = None
    d1_diasbp_max: Optional[float] = None
    d1_diasbp_min: Optional[float] = None
    d1_diasbp_noninvasive_max: Optional[float] = None
    d1_diasbp_noninvasive_min: Optional[float] = None
    d1_heartrate_max: Optional[float] = None
    d1_heartrate_min: Optional[float] = None
    d1_mbp_max: Optional[float] = None
    d1_mbp_min: Optional[float] = None
    d1_mbp_noninvasive_max: Optional[float] = None
    d1_mbp_noninvasive_min: Optional[float] = None
    d1_resprate_max: Optional[float] = None
    d1_resprate_min: Optional[float] = None
    d1_spo2_max: Optional[float] = None
    d1_spo2_min: Optional[float] = None
    d1_sysbp_max: Optional[float] = None
    d1_sysbp_min: Optional[float] = None
    d1_sysbp_noninvasive_max: Optional[float] = None
    d1_sysbp_noninvasive_min: Optional[float] = None
    d1_temp_max: Optional[float] = None
    d1_temp_min: Optional[float] = None
    h1_diasbp_max: Optional[float] = None
    h1_diasbp_min: Optional[float] = None
    h1_diasbp_noninvasive_max: Optional[float] = None
    h1_diasbp_noninvasive_min: Optional[float] = None
    h1_heartrate_max: Optional[float] = None
    h1_heartrate_min: Optional[float] = None
    h1_mbp_max: Optional[float] = None
    h1_mbp_min: Optional[float] = None
    h1_mbp_noninvasive_max: Optional[float] = None
    h1_mbp_noninvasive_min: Optional[float] = None
    h1_resprate_max: Optional[float] = None
    h1_resprate_min: Optional[float] = None
    h1_spo2_max: Optional[float] = None
    h1_spo2_min: Optional[float] = None
    h1_sysbp_max: Optional[float] = None
    h1_sysbp_min: Optional[float] = None
    h1_sysbp_noninvasive_max: Optional[float] = None
    h1_sysbp_noninvasive_min: Optional[float] = None
    d1_glucose_max: Optional[float] = None
    d1_glucose_min: Optional[float] = None
    d1_potassium_max: Optional[float] = None
    d1_potassium_min: Optional[float] = None
    apache_4a_hospital_death_prob: Optional[float] = None
    apache_4a_icu_death_prob: Optional[float] = None
    aids: Optional[bool] = None
    cirrhosis: Optional[bool] = None
    diabetes_mellitus: Optional[bool] = None
    hepatic_failure: Optional[bool] = None
    immunosuppression: Optional[bool] = None
    leukemia: Optional[bool] = None
    lymphoma: Optional[bool] = None
    solid_tumor_with_metastasis: Optional[bool] = None
    apache_3j_bodysystem: Optional[str] = None
    apache_2_bodysystem: Optional[str] = None

@app.post("/predict")
async def predict(data: PatientData):
    return model.predict(data)