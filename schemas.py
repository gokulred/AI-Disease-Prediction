from pydantic import BaseModel, Field
from typing import Literal

#  Diabetes  Schema 
class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, example=6)
    Glucose: float = Field(..., ge=0, example=148)
    BloodPressure: float = Field(..., ge=0, example=72)
    SkinThickness: float = Field(..., ge=0, example=35)
    Insulin: float = Field(..., ge=0, example=0)
    BMI: float = Field(..., ge=0, example=33.6)
    DiabetesPedigreeFunction: float = Field(..., ge=0, example=0.627)
    Age: int = Field(..., ge=0, example=50)

# Heart Disease  Schema 
class HeartInput(BaseModel):
    Age: int = Field(..., ge=0, example=40)
    Sex: Literal['M', 'F'] = Field(..., example='M')
    ChestPainType: Literal['ATA', 'NAP', 'ASY', 'TA'] = Field(..., example='ATA')
    RestingBP: int = Field(..., ge=0, example=140)
    Cholesterol: int = Field(..., ge=0, example=289)
    FastingBS: int = Field(..., description="1 if > 120 mg/dl, else 0", ge=0, le=1, example=0)
    RestingECG: Literal['Normal', 'ST', 'LVH'] = Field(..., example='Normal')
    MaxHR: int = Field(..., ge=0, example=172)
    ExerciseAngina: Literal['N', 'Y'] = Field(..., example='N')
    Oldpeak: float = Field(..., example=0.0)
    ST_Slope: Literal['Up', 'Flat', 'Down'] = Field(..., example='Up')

#  Parkinson'sSchema 
class ParkinsonInput(BaseModel):
    mdvp_fo_hz: float = Field(..., alias="MDVP:Fo(Hz)", example=119.992)
    mdvp_fhi_hz: float = Field(..., alias="MDVP:Fhi(Hz)", example=157.302)
    mdvp_flo_hz: float = Field(..., alias="MDVP:Flo(Hz)", example=74.997)
    mdvp_jitter_percent: float = Field(..., alias="MDVP:Jitter(%)", example=0.00784)
    mdvp_jitter_abs: float = Field(..., alias="MDVP:Jitter(Abs)", example=0.00007)
    mdvp_rap: float = Field(..., alias="MDVP:RAP", example=0.00370)
    mdvp_ppq: float = Field(..., alias="MDVP:PPQ", example=0.00554)
    jitter_ddp: float = Field(..., alias="Jitter:DDP", example=0.01109)
    mdvp_shimmer: float = Field(..., alias="MDVP:Shimmer", example=0.04374)
    mdvp_shimmer_db: float = Field(..., alias="MDVP:Shimmer(dB)", example=0.426)
    shimmer_apq3: float = Field(..., alias="Shimmer:APQ3", example=0.02182)
    shimmer_apq5: float = Field(..., alias="Shimmer:APQ5", example=0.03130)
    mdvp_apq: float = Field(..., alias="MDVP:APQ", example=0.02971)
    shimmer_dda: float = Field(..., alias="Shimmer:DDA", example=0.06545)
    nhr: float = Field(..., alias="NHR", example=0.02211)
    hnr: float = Field(..., alias="HNR", example=21.033)
    rpde: float = Field(..., alias="RPDE", example=0.414783)
    dfa: float = Field(..., alias="DFA", example=0.815285)
    spread1: float = Field(..., alias="spread1", example=-4.813031)
    spread2: float = Field(..., alias="spread2", example=0.266482)
    d2: float = Field(..., alias="D2", example=2.301442)
    ppe: float = Field(..., alias="PPE", example=0.284654)

    class Config:
        populate_by_name = True