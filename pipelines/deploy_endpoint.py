from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration
)
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

with open(".model_version.txt") as f:
    model_version = f.read().strip()

ENDPOINT_NAME = "churn-pred-api"

# Create the endpoint
endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="Real-time customer churn prediction REST API",
    auth_mode="key",
    tags={"project": "churn-mlops"}
)

print("Creating endpoint...")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Endpoint created.")

# Create the deployment
deployment = ManagedOnlineDeployment(
    name="churn-v1",
    endpoint_name=ENDPOINT_NAME,
    model=f"azureml:churn-xgboost-classifier:{model_version}",
    code_configuration=CodeConfiguration(
        code="./src",
        scoring_script="score.py"
    ),
    environment="churn-training-env:2",
    instance_type="Standard_DS3_v2",
    instance_count=1
)

print("Creating deployment — this takes 8–12 minutes...")
ml_client.online_deployments.begin_create_or_update(deployment).result()
print("Deployment created.")

# Route all traffic to churn-v1
endpoint.traffic = {"churn-v1": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Traffic routed to churn-v1.")

# Print URI and key
endpoint_info = ml_client.online_endpoints.get(ENDPOINT_NAME)
keys          = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)

print(f"\n{'='*60}")
print(f"Endpoint URI : {endpoint_info.scoring_uri}")
print(f"Primary Key  : {keys.primary_key}")
print(f"{'='*60}")

with open(".endpoint_info.txt", "w") as f:
    f.write(f"{endpoint_info.scoring_uri}\n{keys.primary_key}")