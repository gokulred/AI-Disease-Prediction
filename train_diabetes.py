import pandas as pd
import pickle
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import shap

DATA_PATH = "datasets/diabetes.csv"
EXPERIMENT_NAME = "Diabetes_Risk_Prediction"

def train():
   
    os.makedirs("models", exist_ok=True)

    print(f"Loading data from {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Error: Dataset not found")
        return

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        print("Starting training")

        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42
        }
        mlflow.log_params(params)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(**params))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

        print("Saving model locally")
        with open("models/diabetes_model.pkl", "wb") as f:
            pickle.dump(pipeline, f)
        print("Model saved to models/diabetes_model.pkl")

        print("Generating SHAP explainer")
        model = pipeline.named_steps['classifier']
        
      
        preprocessor = pipeline.named_steps['scaler']
        X_train_transformed = preprocessor.transform(X_train)
        
        explainer = shap.TreeExplainer(model)
        
        with open("models/diabetes_shap_explainer.pkl", "wb") as f:
            pickle.dump(explainer, f)
        mlflow.log_artifact("models/diabetes_shap_explainer.pkl")

        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="diabetes_pipeline",
            signature=signature,
            input_example=X_train.iloc[:5]
        )

        print("Training complete.")

if __name__ == "__main__":
    train()