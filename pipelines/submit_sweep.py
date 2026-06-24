from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.sweep import Choice, Uniform
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

# ── Step 1: Define the base command (same as baseline, but params are sweep vars) ──
base_job = command(
    display_name="churn-sweep-trial",
    code="./src",
    command=(
        "python train.py "
        "--data_path ${{inputs.training_data}} "
        "--learning_rate ${{search_space.learning_rate}} "
        "--max_depth ${{search_space.max_depth}} "
        "--n_estimators ${{search_space.n_estimators}} "
        "--subsample ${{search_space.subsample}} "
        "--colsample_bytree ${{search_space.colsample_bytree}} "
        "--model_output ${{outputs.model_output}}"
    ),
    inputs={
        "training_data": Input(
            type=AssetTypes.URI_FILE,
            path="azureml:telco-churn-raw:1"
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
    experiment_name="churn-mlops-experiments"
)

# ── Step 2: Convert to sweep job ───────────────────────────────────────────────────
sweep_job = base_job.sweep(
    sampling_algorithm="random",   # Random search: fast and surprisingly effective
    primary_metric="roc_auc",      # Must match mlflow.log_metric("roc_auc") in train.py exactly
    goal="Maximize",
    search_space={
        "learning_rate":    Uniform(min_value=0.01, max_value=0.30),
        "max_depth":        Choice([3, 4, 5, 6, 8]),
        "n_estimators":     Choice([100, 150, 200, 300]),
        "subsample":        Uniform(min_value=0.6,  max_value=1.0),
        "colsample_bytree": Uniform(min_value=0.6,  max_value=1.0)
    }
)

# ── Step 3: Set resource limits ────────────────────────────────────────────────────
sweep_job.set_limits(
    max_total_trials=20,       # Total hyperparameter combinations to try
    max_concurrent_trials=2,   # Run 4 trials simultaneously (matches cluster max-instances)
    timeout=7200               # Hard stop at 2 hours regardless
)

sweep_job.display_name = "churn-xgboost-hyperdrive-sweep"
sweep_job.tags = {"project": "churn-mlops", "run_type": "hyperdrive_sweep"}

returned_sweep = ml_client.jobs.create_or_update(sweep_job)
print(f"Sweep submitted: {returned_sweep.name}")
print(f"Studio URL:      {returned_sweep.studio_url}")

# Save the sweep job name — register_model.py needs it
with open(".sweep_job_name.txt", "w") as f:
    f.write(returned_sweep.name)