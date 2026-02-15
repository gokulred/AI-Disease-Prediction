import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "datasets/heart.csv"
EXPERIMENT_NAME = "Heart_Disease_Prediction"

def train():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)

        X = df.drop(['HeartDisease'], axis=1)
        y = df['HeartDisease']

        X = pd.get_dummies(X)
        
        model_columns = list(X.columns)
        with open('models/heart_columns.pkl', 'wb') as f:
            pickle.dump(model_columns, f)
        print("Saved feature columns.")

       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        scaler = StandardScaler()
        
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        with open('models/heart_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Saved scaler.")

        params = {"n_estimators": 100, "random_state": 42}
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Heart Model - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        mlflow.log_metrics({"accuracy": acc, "f1": f1})
        mlflow.log_params(params)

        with open('models/heart_model.pkl', 'wb') as f:
            pickle.dump(rf, f)
        
        mlflow.sklearn.log_model(rf, "heart_rf_model")
        print("Heart training complete.")

if __name__ == "__main__":
    train()