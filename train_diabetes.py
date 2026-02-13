import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import shap

DATA_PATH = "datasets/diabetes.csv"
EXPERIMENT_NAME = "Diabetes_Risk_Prediction"

def train():

    #load data

    print(f"Loading data from {DATA_PATH}")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Error: Dataset not found")

    #split
    
    X = df.drop('Outcome',axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    mlflow.set_experiment(EXPERIMENT_NAME)

    #params

    with mlflow.start_run():
        print("Starting training")

        params = {
            "n_estimators":200,
            "max_depth":10,
            "min_samples_split":2,
            "random_state":42
        }

        mlflow.log_params(params)

        #train model

        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        #evaluate

        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f},F1 Score:{f1:.4f}")

        mlflow.log_metrics({"accuracy": accuracy,"f1_score":f1})

        print("SHAP explainer generating")
        explainer = shap.TreeExplainer(rf)

        import pickle
        with open("models/diabetes_shap_explainer.pkl", "wb") as f:
            pickle.dump(explainer, f)
        mlflow.log_artifact("models/diabetes_shap_explainer.pkl")

        signature = infer_signature(X_train, rf.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model = rf,
            artifact_path = "random_forest_model",
            signature = signature,
            input_example = X_train.iloc[:5]
        )

        print ("Trainig complete")

if __name__ == "__main__":
    train()





      


        

      
     

    

