import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import os

DATASET_PATH = "weatherAUS_uluru_preprocessed.csv"


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except:
        df = pd.read_csv(f"./MLProject/{filepath}")
    return df


def prepare_data(df):
    X = df.drop(columns=["RainTomorrow"])
    y = df["RainTomorrow"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_model(
    X_train,
    X_test,
    y_train,
    y_test
):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("weatherAUS_uluru_RF")


    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.sklearn.autolog()
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
            
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "random_forest_model")

        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        with open("run_id.txt", "w") as f:
            f.write(run_id)


def main():
    print("Loading dataset...")

    df = load_data(DATASET_PATH)

    print(f"Dataset shape: {df.shape}")

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_data(df)

    print("Training Random Forest...")

    train_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Training completed.")

main()