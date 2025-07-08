# test_api.py
import requests
import json

def test_api():
    """Test the readmission prediction API"""
    
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.json()}")
    
    # Get example patient
    print("\nGetting example patient...")
    response = requests.get(f"{base_url}/example_patient")
    example_patient = response.json()
    print(f"Example patient: {json.dumps(example_patient, indent=2)}")
    
    # Test prediction
    print("\nTesting prediction...")
    response = requests.post(f"{base_url}/predict", json=example_patient)
    prediction = response.json()
    print(f"Prediction result: {json.dumps(prediction, indent=2)}")
    
    # Test with different patient
    high_risk_patient = {
        "patient_id": "P000002",
        "age": 85,
        "gender": "M",
        "race": "Black",
        "insurance": "Medicaid",
        "length_of_stay": 12,
        "diabetes": 1,
        "heart_failure": 1,
        "copd": 1,
        "hypertension": 1,
        "charlson_score": 6,
        "prior_admissions_1yr": 4,
        "ed_visits_6m": 3,
        "num_medications": 15,
        "lives_alone": 1,
        "transportation_issues": 1,
        "discharge_disposition": "Home"
    }
    
    print("\nTesting high-risk patient...")
    response = requests.post(f"{base_url}/predict", json=high_risk_patient)
    prediction = response.json()
    print(f"High-risk prediction: {json.dumps(prediction, indent=2)}")

if __name__ == "__main__":
    test_api()
