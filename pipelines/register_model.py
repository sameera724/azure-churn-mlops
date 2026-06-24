from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
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

best_run_name = "witty_carnival_0msnf7x24g_17"
best_roc_auc  = 0.8460
best_f1       = 0.5945 

best_job = ml_client.jobs.get(best_run_name)
print(f"Best run: {best_job.name}")

model = Model(
    name="churn-xgboost-classifier",
    description=(
        f"XGBoost customer churn classifier. "
        f"Dataset: IBM Telco (7,043 rows, 21 features). "
        f"Tuned via 20-trial HyperDrive random sweep. "
        f"Best ROC-AUC: {best_roc_auc}  F1: {best_f1}."
    ),
    path=f"azureml://jobs/{best_job.name}/outputs/model_output/",
    type=AssetTypes.CUSTOM_MODEL,
    tags={
        "roc_auc":      str(round(best_roc_auc, 4)),
        "f1_score":     str(round(best_f1, 4)),
        "framework":    "xgboost-1.7.6",
        "dataset":      "telco-customer-churn-v1",
        "project":      "churn-mlops",
        "training_job": best_job.name
    }
)

registered = ml_client.models.create_or_update(model)
print(f"\nModel registered: {registered.name}  version: {registered.version}")

with open(".model_version.txt", "w") as f:
    f.write(str(registered.version))