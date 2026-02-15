import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "datasets/parkinsons.csv"
EXPERIMENT_NAME = "Parkinsons_Prediction"

def train():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)

        X = df.drop(['name', 'status'], axis=1)
        y = df['status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Parkinsons Model - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        with open('models/parkinsons_pipeline.pkl', 'wb') as f:
            pickle.dump(pipeline, f)

        mlflow.sklearn.log_model(pipeline, "parkinsons_rf_pipeline")
        print("Parkinsons training complete. Pipeline saved.")

if __name__ == "__main__":
    train()