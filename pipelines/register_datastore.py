from azure.ai.ml import MLClient
from azure.ai.ml.entities import AzureBlobDatastore, AccountKeyConfiguration
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

datastore = AzureBlobDatastore(
    name="adls_churn",
    description="ADLS Gen2 data lake for churn MLOps project",
    account_name="sadlchurnmlops",
    container_name="raw",
    credentials=AccountKeyConfiguration(
        account_key=os.environ["ACCOUNT_KEY"]
    )
)

registered = ml_client.datastores.create_or_update(datastore)
print(f"Datastore registered: {registered.name}")
