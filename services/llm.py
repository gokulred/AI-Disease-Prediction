import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_health_report(disease_name: str, prediction: int, probability: float, input_data: dict):
    """
    Generates a personalized health report using GenAI.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
      
        risk_status = "HIGH RISK" if prediction == 1 else "Low Risk"
        
        
        prompt = f"""
        You are an empathetic medical AI assistant. 
        A machine learning model has analyzed a patient's data for {disease_name}.
        
        Model Results:
        - Prediction: {risk_status}
        - Probability: {probability:.2%}
        
        Patient Data:
        {input_data}
        
         Task:
        1. Explain the result to the patient in simple, calm language.
        2. Highlight 1-2 key risk factors from the data provided (e.g., if Glucose is high, mention it).
        3. Suggest 2-3 general lifestyle tips suitable for this condition.
        4. Disclaimer: End with a strong note that this is an AI prediction, not a doctor's diagnosis.
        
        Keep the response short (under 100 words).
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        return f"AI analysis unavailable at the moment. (Error: {str(e)})"