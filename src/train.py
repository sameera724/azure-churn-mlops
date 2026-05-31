import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier
import joblib


def load_and_clean(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(0, inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    return df


def build_pipeline(numerical_cols: list, categorical_cols: list, params: dict) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ])
    classifier = XGBClassifier(
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",        type=str,   required=True)
    parser.add_argument("--learning_rate",    type=float, default=0.1)
    parser.add_argument("--max_depth",        type=int,   default=6)
    parser.add_argument("--n_estimators",     type=int,   default=100)
    parser.add_argument("--subsample",        type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--model_output",     type=str,   required=True)
    args = parser.parse_args()

    # Azure ML automatically connects MLflow to the workspace — no config needed
    with mlflow.start_run():

        params = {
            "learning_rate":    args.learning_rate,
            "max_depth":        args.max_depth,
            "n_estimators":     args.n_estimators,
            "subsample":        args.subsample,
            "colsample_bytree": args.colsample_bytree,
        }
        mlflow.log_params(params)

        # ── Load and split ────────────────────────────────────────────────────
        df = load_and_clean(args.data_path)
        X  = df.drop(columns=["Churn"])
        y  = df["Churn"]

        numerical_cols   = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # ── Train ─────────────────────────────────────────────────────────────
        pipeline = build_pipeline(numerical_cols, categorical_cols, params)
        pipeline.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)
        f1      = f1_score(y_test, y_pred)

        # IMPORTANT: metric name must match HyperDrive primary_metric exactly
        mlflow.log_metric("roc_auc",  roc_auc)
        mlflow.log_metric("f1_score", f1)

        print(f"ROC-AUC : {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\n" + classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

        # ── Feature importance ────────────────────────────────────────────────
        cat_feature_names = list(
            pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .get_feature_names_out(categorical_cols)
        )
        all_feature_names = numerical_cols + cat_feature_names
        importances = pipeline.named_steps["classifier"].feature_importances_

        importance_df = pd.DataFrame({
            "feature":    all_feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(20)

        importance_path = os.path.join(args.model_output, "feature_importance.csv")
        os.makedirs(args.model_output, exist_ok=True)
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # ── Save model ────────────────────────────────────────────────────────
        model_path = os.path.join(args.model_output, "model.pkl")
        joblib.dump(pipeline, model_path)

        # Log as MLflow model — enables automatic batch inference without score.py
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="churn_model",
            registered_model_name=None  # We register manually for control
        )

        print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()