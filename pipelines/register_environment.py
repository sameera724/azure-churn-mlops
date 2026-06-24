from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
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

env = Environment(
    name="churn-training-env",
    description="XGBoost + scikit-learn environment for churn MLOps project",
    conda_file="environment/conda_dependencies.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

registered = ml_client.environments.create_or_update(env)
print(f"Registered environment: {registered.name}  version: {registered.version}")