
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schemas import DiabetesInput, HeartInput, ParkinsonInput

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
  
    try:
        print("Loading models...")
        ml_models["diabetes_model"] = pickle.load(open('models/diabetes_model.pkl', 'rb'))
        
        ml_models["heart_model"] = pickle.load(open('models/heart_model.pkl', 'rb'))
        ml_models["heart_scaler"] = pickle.load(open('models/heart_scaler.pkl', 'rb'))
        ml_models["heart_columns"] = pickle.load(open('models/heart_columns.pkl', 'rb'))
        
        ml_models["parkinsons_model"] = pickle.load(open('models/parkinsons_model.pkl', 'rb'))
        ml_models["parkinsons_scaler"] = pickle.load(open('models/parkinsons_scaler.pkl', 'rb'))
        ml_models["parkinsons_columns"] = pickle.load(open('models/parkinsons_columns.pkl', 'rb'))
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
    
    yield
    ml_models.clear()

app = FastAPI(title="Multi-Disease Prediction API", version="1.0", lifespan=lifespan)


def get_risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

@app.get("/")
def home():
    return {"message": "API is running. Use /docs for the Swagger UI."}

#diabetes

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    model = ml_models.get("diabetes_model")
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
  
    input_data = data.model_dump()
    input_df = pd.DataFrame([input_data])
    
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    
    return {
        "prediction": int(pred),
        "probability": round(prob, 3),
        "risk_level": get_risk_level(prob)
    }

#heart

@app.post("/predict/heart")
def predict_heart(data: HeartInput):

    pipeline = ml_models.get("heart_pipeline")

    
    if not pipeline:
        raise HTTPException(status_code = 500, detail = "Heart model pipeline not loaded")
   
  
    input_df = pd.DataFrame([data.model_dump()])

    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    return {
        "prediction": int(pred),
        "probability": round(prob, 3),
        "risk_level": get_risk_level(prob)
    }

#parkinsons

@app.post("/predict/parkinsons")
def predict_parkinsons(data: ParkinsonInput):

    pipeline = ml_models.get("parkinsons_pipeline")

    if not pipeline:
        raise HTTPException(status_code = 500, detail = "Parkinson's model pipeline not loaded")
    
    input_data = data.model_dump(by_alias = True)
    input_df  = pd.DataFrame([input_data])

    prob = pipeline.predict_proba(input_df)[0][1]
    pred = pipeline.predict(input_df)[0]
   

    return {
        "prediction": int(pred),
        "probability": round(prob, 3),
        "risk_level": get_risk_level(prob)
    }