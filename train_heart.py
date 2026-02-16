import pandas as pd
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

DATA_PATH = "datasets/heart.csv"
EXPERIMENT_NAME = "Heart_Disease_Prediction"
MODEL_PATH = "models/heart_pipeline.pkl"

def train():
    os.makedirs("models", exist_ok=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(['HeartDisease'], axis=1)
    y = df['HeartDisease']

    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        print("Starting training")
        pipeline.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, pipeline.predict(X_test))
        print(f"Accuracy: {accuracy:.4f}")
        mlflow.log_metric("accuracy", accuracy)

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(pipeline, f)
        
        mlflow.sklearn.log_model(pipeline, "heart_pipeline")
        print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()