import os
import json
import logging
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def init():
    """Called once when the endpoint container starts. Load model into memory."""
    global model

    # AZUREML_MODEL_DIR is injected by Azure ML at runtime
    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model.pkl")

    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully.")


def run(raw_data: str) -> str:
    """Called for every inference request. raw_data is a JSON string."""
    try:
        payload   = json.loads(raw_data)
        records   = payload.get("data", payload)  # Accept {"data": [...]} or raw list
        input_df  = pd.DataFrame(records)

        predictions   = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]

        result = {
            "predictions":       [int(p)              for p in predictions],
            "churn_probability": [round(float(p), 4)  for p in probabilities],
            "churn_label":       ["Yes" if p == 1 else "No" for p in predictions]
        }
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Scoring error: {e}")
        return json.dumps({"error": str(e)})