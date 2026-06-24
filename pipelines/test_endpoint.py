import requests
import json

BASE_URL = "http://churn-prediction-api.northcentralus.azurecontainer.io:8000"

# Health check
health = requests.get(f"{BASE_URL}/health")
print(f"Health check: {health.json()}")

# Score two customers
sample_data = {
    "data": [
        {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "No", "tenure": 1, "PhoneService": "No",
            "MultipleLines": "No phone service", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85, "TotalCharges": 29.85
        },
        {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
            "Dependents": "No", "tenure": 72, "PhoneService": "Yes",
            "MultipleLines": "Yes", "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "Yes", "TechSupport": "Yes",
            "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Two year", "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 95.70, "TotalCharges": 6888.40
        }
    ]
}

response = requests.post(f"{BASE_URL}/score", json=sample_data)
print(f"\nStatus: {response.status_code}")
print(f"Response:\n{json.dumps(response.json(), indent=2)}")