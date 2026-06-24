from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

load_dotenv()

ml_client = MLClient(
    DefaultAzureCredential(),
    os.environ["SUBSCRIPTION_ID"],
    os.environ["RESOURCE_GROUP"],
    os.environ["WORKSPACE_NAME"]
)

job = command(
    display_name="churn-xgboost-baseline",
    description="Baseline XGBoost run — default hyperparams — verify pipeline works",

    # Path to local source code. AML zips and uploads this automatically.
    code="./src",

    # ${{inputs.X}} and ${{outputs.X}} are substituted at runtime by AML
    command=(
        "python train.py "
        "--data_path ${{inputs.training_data}} "
        "--learning_rate 0.1 "
        "--max_depth 6 "
        "--n_estimators 100 "
        "--subsample 0.8 "
        "--colsample_bytree 0.8 "
        "--model_output ${{outputs.model_output}}"
    ),

    inputs={
        "training_data": Input(
            type=AssetTypes.URI_FILE,
            path="azureml:telco-churn-raw:1"   # references your registered data asset
        )
    },
    outputs={
        "model_output": Output(
            type=AssetTypes.URI_FOLDER,
            mode=InputOutputModes.RW_MOUNT
        )
    },

    environment="churn-training-env:2",
    compute="cpu-cluster",

    experiment_name="churn-mlops-experiments",
    tags={"project": "churn-mlops", "run_type": "baseline"}
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Studio URL:    {returned_job.studio_url}")