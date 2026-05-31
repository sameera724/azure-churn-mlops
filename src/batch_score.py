import os
import logging
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def init():
    """Called once per node when batch job starts."""
    global model

    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    logger.info("Batch scoring model loaded.")


def run(mini_batch: list) -> pd.DataFrame:
    """
    Called with a mini-batch of file paths. Each file is a CSV chunk.
    Must return a DataFrame — Azure ML appends all results to the output file.
    """
    results = []

    for file_path in mini_batch:
        df = pd.read_csv(file_path)

        # Preserve any ID column if present
        id_col = df["customerID"] if "customerID" in df.columns else pd.Series(range(len(df)))
        df_features = df.drop(columns=["customerID"], errors="ignore")
        df_features = df_features.drop(columns=["Churn"], errors="ignore")

        predictions   = model.predict(df_features)
        probabilities = model.predict_proba(df_features)[:, 1]

        result_df = pd.DataFrame({
            "customerID":        id_col.values,
            "prediction":        predictions,
            "churn_probability": [round(float(p), 4) for p in probabilities],
            "churn_label":       ["Yes" if p == 1 else "No" for p in predictions],
            "source_file":       os.path.basename(file_path)
        })
        results.append(result_df)

    return pd.concat(results, ignore_index=True)