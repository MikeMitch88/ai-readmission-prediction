# api.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from data_preprocessing import ReadmissionPreprocessor

app = Flask(__name__)

# Load model and preprocessor at startup
print("Loading model and preprocessor...")
model_data = joblib.load('readmission_model.pkl')
model = model_data['model']
preprocessor = ReadmissionPreprocessor.load('preprocessor.pkl')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": model_data['model_name']})

@app.route('/predict', methods=['POST'])
def predict_readmission():
    """Predict readmission risk for a patient"""
    try:
        # Get patient data
        patient_data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Make prediction
        probability = model.predict_proba(X)[0][1]
        prediction = model.predict(X)[0]
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Get feature importance for explanation
        feature_importance = get_patient_risk_factors(patient_data, X[0])
        
        response = {
            "patient_id": patient_data.get("patient_id", "unknown"),
            "readmission_probability": float(probability),
            "predicted_readmission": bool(prediction),
            "risk_level": risk_level,
            "top_risk_factors": feature_importance[:5],
            "recommendations": get_recommendations(risk_level, patient_data)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def get_patient_risk_factors(patient_data, processed_features):
    """Get top risk factors for a specific patient"""
    # Get feature importance from model
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return []
    
    # Combine with patient's feature values
    risk_factors = []
    for i, (feature_name, feature_importance) in enumerate(zip(preprocessor.feature_names, importance)):
        if i < len(processed_features):
            risk_score = feature_importance * abs(processed_features[i])
            risk_factors.append({
                "factor": feature_name,
                "importance": float(feature_importance),
                "patient_value": float(processed_features[i]),
                "risk_score": float(risk_score)
            })
    
    # Sort by risk score
    risk_factors.sort(key=lambda x: x['risk_score'], reverse=True)
    return risk_factors

def get_recommendations(risk_level, patient_data):
    """Generate recommendations based on risk level and patient characteristics"""
    recommendations = []
    
    if risk_level == "High":
        recommendations.extend([
            "Schedule follow-up appointment within 7 days",
            "Consider discharge planning consultation",
            "Ensure medication reconciliation is completed",
            "Arrange home health services if appropriate"
        ])
    elif risk_level == "Medium":
        recommendations.extend([
            "Schedule follow-up appointment within 14 days",
            "Review discharge instructions with patient",
            "Consider telehealth check-in within 3 days"
        ])
    else:
        recommendations.append("Standard discharge planning appropriate")
    
    # Add specific recommendations based on patient characteristics
    if patient_data.get('age', 0) > 75:
        recommendations.append("Consider geriatric assessment")
    
    if patient_data.get('lives_alone', 0) == 1:
        recommendations.append("Assess social support and home safety")
    
    if patient_data.get('num_medications', 0) > 10:
        recommendations.append("Pharmacy consultation for medication management")
    
    if patient_data.get('charlson_score', 0) > 3:
        recommendations.append("Consider care coordination with primary care")
    
    return recommendations

# Example patient data endpoint for testing
@app.route('/example_patient', methods=['GET'])
def get_example_patient():
    """Get example patient data for testing"""
    example = {
        "patient_id": "P000001",
        "age": 72,
        "gender": "F",
        "race": "White",
        "insurance": "Medicare",
        "length_of_stay": 5,
        "diabetes": 1,
        "heart_failure": 1,
        "copd": 0,
        "hypertension": 1,
        "charlson_score": 4,
        "prior_admissions_1yr": 2,
        "ed_visits_6m": 1,
        "num_medications": 12,
        "lives_alone": 1,
        "transportation_issues": 0,
        "discharge_disposition": "Home"
    }
    return jsonify(example)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
