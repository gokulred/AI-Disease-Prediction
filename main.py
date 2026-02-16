import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schemas import DiabetesInput, HeartInput, ParkinsonInput
from services.llm import generate_health_report
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading models...")
    
    # 1. Diabetes Pipeline
    with open('models/diabetes_model.pkl', 'rb') as f:
        ml_models["diabetes_model"] = pickle.load(f)
        
    # 2. Heart Disease  Pipeline 
    with open('models/heart_pipeline.pkl', 'rb') as f:
        ml_models["heart_pipeline"] = pickle.load(f)

    # Parkinsons Pipeline
    with open('models/parkinsons_pipeline.pkl', 'rb') as f:
        ml_models["parkinsons_pipeline"] = pickle.load(f)
        
    print("All models loaded successfully.")
    yield
    ml_models.clear()

app = FastAPI(title="Multi-Disease Prediction API", version="2.0", lifespan=lifespan)

def get_risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

@app.get("/")
def home():
    return {"message": "Health AI Prediction Service is Running."}

@app.post("/predict/diabetes", tags=["Diabetes"])
def predict_diabetes(data: DiabetesInput):
    model = ml_models.get("diabetes_model")
    if not model:
        raise HTTPException(status_code=503, detail="Diabetes model is not available")
  
    input_data = data.model_dump()
    input_df = pd.DataFrame([input_data])
    
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    
    ai_explanation = generate_health_report(
        disease_name="Diabetes",
        prediction=int(pred),
        probability=prob,
        input_data=input_data
    )

    return {
        "prediction": int(pred),
        "probability": round(prob, 3),
        "risk_level": get_risk_level(prob),
        "ai_analysis": ai_explanation
    }

@app.post("/predict/heart", tags=["Heart Disease"])
def predict_heart(data: HeartInput):
    pipeline = ml_models.get("heart_pipeline")

    if not pipeline:
        raise HTTPException(status_code=503, detail="Heart model pipeline is not available")

    input_data = data.model_dump()
   
    input_df = pd.DataFrame([input_data])

    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    ai_explanation = generate_health_report(
        disease_name="Heart Disease",
        prediction=int(pred),
        probability=prob,
        input_data=input_data
    )

    return {
        "prediction": int(pred),
        "probability": round(prob, 3),
        "risk_level": get_risk_level(prob),
        "ai_analysis": ai_explanation
    }

@app.post("/predict/parkinsons", tags=["Parkinsons"])
def predict_parkinsons(data: ParkinsonInput):
    pipeline = ml_models.get("parkinsons_pipeline")
    if not pipeline:
        raise HTTPException(status_code=503, detail="Parkinson's model pipeline is not available")
    
    
    input_data = data.model_dump(by_alias=True)
    input_df  = pd.DataFrame([input_data])

    prob = pipeline.predict_proba(input_df)[0][1]
    pred = pipeline.predict(input_df)[0]

    ai_explanation = generate_health_report(
        disease_name="Parkinson's",
        prediction=int(pred),
        probability=prob,
        input_data=input_data
    )

    return {
        "prediction": int(pred),
        "probability": round(prob, 3),
        "risk_level": get_risk_level(prob),
        "ai_analysis": ai_explanation

    }