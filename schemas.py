from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

#  Diabetes Schema
class DiabetesInput(BaseModel):
   
    model_config = ConfigDict(populate_by_name=True)

    Pregnancies: int = Field(..., ge=0, json_schema_extra={"example": 6})
    Glucose: float = Field(..., ge=0, json_schema_extra={"example": 148})
    BloodPressure: float = Field(..., ge=0, json_schema_extra={"example": 72})
    SkinThickness: float = Field(..., ge=0, json_schema_extra={"example": 35})
    Insulin: float = Field(..., ge=0, json_schema_extra={"example": 0})
    BMI: float = Field(..., ge=0, json_schema_extra={"example": 33.6})
    DiabetesPedigreeFunction: float = Field(..., ge=0, json_schema_extra={"example": 0.627})
    Age: int = Field(..., ge=0, json_schema_extra={"example": 50})

#  Heart Disease Schema
class HeartInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Age: int = Field(..., ge=0, json_schema_extra={"example": 40})
    Sex: Literal['M', 'F'] = Field(..., json_schema_extra={"example": 'M'})
    ChestPainType: Literal['ATA', 'NAP', 'ASY', 'TA'] = Field(..., json_schema_extra={"example": 'ATA'})
    RestingBP: int = Field(..., ge=0, json_schema_extra={"example": 140})
    Cholesterol: int = Field(..., ge=0, json_schema_extra={"example": 289})
    FastingBS: int = Field(..., ge=0, le=1, description="1 if > 120 mg/dl, else 0", json_schema_extra={"example": 0})
    RestingECG: Literal['Normal', 'ST', 'LVH'] = Field(..., json_schema_extra={"example": 'Normal'})
    MaxHR: int = Field(..., ge=0, json_schema_extra={"example": 172})
    ExerciseAngina: Literal['N', 'Y'] = Field(..., json_schema_extra={"example": 'N'})
    Oldpeak: float = Field(..., json_schema_extra={"example": 0.0})
    ST_Slope: Literal['Up', 'Flat', 'Down'] = Field(..., json_schema_extra={"example": 'Up'})

# Parkinson's Schema
class ParkinsonInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    mdvp_fo_hz: float = Field(..., alias="MDVP:Fo(Hz)", json_schema_extra={"example": 119.992})
    mdvp_fhi_hz: float = Field(..., alias="MDVP:Fhi(Hz)", json_schema_extra={"example": 157.302})
    mdvp_flo_hz: float = Field(..., alias="MDVP:Flo(Hz)", json_schema_extra={"example": 74.997})
    mdvp_jitter_percent: float = Field(..., alias="MDVP:Jitter(%)", json_schema_extra={"example": 0.00784})
    mdvp_jitter_abs: float = Field(..., alias="MDVP:Jitter(Abs)", json_schema_extra={"example": 0.00007})
    mdvp_rap: float = Field(..., alias="MDVP:RAP", json_schema_extra={"example": 0.00370})
    mdvp_ppq: float = Field(..., alias="MDVP:PPQ", json_schema_extra={"example": 0.00554})
    jitter_ddp: float = Field(..., alias="Jitter:DDP", json_schema_extra={"example": 0.01109})
    mdvp_shimmer: float = Field(..., alias="MDVP:Shimmer", json_schema_extra={"example": 0.04374})
    mdvp_shimmer_db: float = Field(..., alias="MDVP:Shimmer(dB)", json_schema_extra={"example": 0.426})
    shimmer_apq3: float = Field(..., alias="Shimmer:APQ3", json_schema_extra={"example": 0.02182})
    shimmer_apq5: float = Field(..., alias="Shimmer:APQ5", json_schema_extra={"example": 0.03130})
    mdvp_apq: float = Field(..., alias="MDVP:APQ", json_schema_extra={"example": 0.02971})
    shimmer_dda: float = Field(..., alias="Shimmer:DDA", json_schema_extra={"example": 0.06545})
    nhr: float = Field(..., alias="NHR", json_schema_extra={"example": 0.02211})
    hnr: float = Field(..., alias="HNR", json_schema_extra={"example": 21.033})
    rpde: float = Field(..., alias="RPDE", json_schema_extra={"example": 0.414783})
    dfa: float = Field(..., alias="DFA", json_schema_extra={"example": 0.815285})
    spread1: float = Field(..., alias="spread1", json_schema_extra={"example": -4.813031})
    spread2: float = Field(..., alias="spread2", json_schema_extra={"example": 0.266482})
    d2: float = Field(..., alias="D2", json_schema_extra={"example": 2.301442})
    ppe: float = Field(..., alias="PPE", json_schema_extra={"example": 0.284654})