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

DATASET_PATH = "weatherAUS_uluru_preprocessed.csv"


def load_data(filepath):
    return pd.read_csv(filepath)


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
    # mlflow.set_tracking_uri("file:./mlruns")
    # mlflow.set_experiment("weatherAUS_Uluru_RF")

    mlflow.sklearn.autolog()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # mlflow.sklearn.log_model(
    #     model,
    #     "random_forest_model"
    # )
    
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