from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment.")

client = genai.Client(api_key=api_key)

def generate_health_report(disease_name, prediction, probability, input_data):
    risk_status = "HIGH RISK" if prediction == 1 else "Low Risk"

    prompt = f"""
    ML result for {disease_name}:
    Prediction: {risk_status}
    Probability: {probability:.2%}
    Data: {input_data}

    Explain simply. Mention 1–2 risk factors.
    Give 2 lifestyle tips.
    Add medical disclaimer.
    Under 100 words.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"LLM ERROR: {str(e)}"
    

