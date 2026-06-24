from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
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

data_asset = Data(
    name="telco-churn-raw",
    version="1",
    description="IBM Telco Customer Churn — 7,043 rows, 21 features, binary classification",
    path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv",  # local file — AML uploads it
    type=AssetTypes.URI_FILE,
    tags={"source": "kaggle-ibm-telco", "rows": "7043", "features": "21"}
)

registered = ml_client.data.create_or_update(data_asset)
print(f"Data asset: {registered.name}  version: {registered.version}")
print(f"Path in AML: {registered.path}")