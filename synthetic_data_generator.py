# synthetic_data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_patient_data(n_patients=10000):
    """Generate synthetic patient data for readmission prediction"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Generate patient demographics
    patients = []
    
    for i in range(n_patients):
        # Basic demographics
        age = np.random.normal(65, 15)
        age = max(18, min(95, age))  # Constrain age
        
        gender = np.random.choice(['M', 'F'], p=[0.45, 0.55])
        race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                               p=[0.6, 0.2, 0.15, 0.03, 0.02])
        
        # Insurance type affects readmission risk
        insurance = np.random.choice(['Medicare', 'Medicaid', 'Private', 'Uninsured'], 
                                   p=[0.4, 0.25, 0.3, 0.05])
        
        # Generate admission details
        admission_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
        length_of_stay = max(1, np.random.poisson(5))
        discharge_date = admission_date + timedelta(days=length_of_stay)
        
        # Generate clinical features
        # Comorbidities (higher age = more comorbidities)
        diabetes = 1 if np.random.random() < (0.1 + age/200) else 0
        heart_failure = 1 if np.random.random() < (0.05 + age/300) else 0
        copd = 1 if np.random.random() < (0.08 + age/250) else 0
        hypertension = 1 if np.random.random() < (0.2 + age/150) else 0
        
        # Calculate Charlson Comorbidity Index
        charlson_score = diabetes + heart_failure + copd + hypertension
        if age > 70:
            charlson_score += 1
        
        # Prior healthcare utilization
        prior_admissions = np.random.poisson(charlson_score * 0.5)
        ed_visits_6m = np.random.poisson(charlson_score * 0.3)
        
        # Medications (polypharmacy increases readmission risk)
        num_medications = max(0, np.random.poisson(3 + charlson_score))
        
        # Social determinants
        lives_alone = 1 if np.random.random() < 0.3 else 0
        transportation_issues = 1 if np.random.random() < 0.2 else 0
        
        # Discharge factors
        discharge_disposition = np.random.choice(['Home', 'SNF', 'Rehab', 'Home_Health'], 
                                               p=[0.6, 0.2, 0.1, 0.1])
        
        # Calculate readmission probability based on risk factors
        readmission_prob = 0.1  # Base rate
        
        # Age risk
        if age > 75:
            readmission_prob += 0.05
        elif age > 65:
            readmission_prob += 0.02
            
        # Comorbidity risk
        readmission_prob += charlson_score * 0.03
        
        # Prior utilization risk
        readmission_prob += prior_admissions * 0.02
        readmission_prob += ed_visits_6m * 0.01
        
        # Medication risk
        if num_medications > 10:
            readmission_prob += 0.04
        elif num_medications > 5:
            readmission_prob += 0.02
            
        # Social risk
        readmission_prob += lives_alone * 0.03
        readmission_prob += transportation_issues * 0.02
        
        # Insurance risk
        if insurance == 'Medicaid':
            readmission_prob += 0.02
        elif insurance == 'Uninsured':
            readmission_prob += 0.05
            
        # Length of stay risk (very short or very long stays)
        if length_of_stay <= 1:
            readmission_prob += 0.03
        elif length_of_stay > 10:
            readmission_prob += 0.02
            
        # Discharge disposition risk
        if discharge_disposition == 'Home' and charlson_score > 2:
            readmission_prob += 0.03
            
        # Cap probability
        readmission_prob = min(0.8, readmission_prob)
        
        # Generate actual readmission outcome
        readmitted_30d = 1 if np.random.random() < readmission_prob else 0
        
        # Add some noise to make it more realistic
        if np.random.random() < 0.05:  # 5% random noise
            readmitted_30d = 1 - readmitted_30d
        
        patient = {
            'patient_id': f'P{i+1:06d}',
            'age': int(age),
            'gender': gender,
            'race': race,
            'insurance': insurance,
            'admission_date': admission_date.strftime('%Y-%m-%d'),
            'discharge_date': discharge_date.strftime('%Y-%m-%d'),
            'length_of_stay': length_of_stay,
            'diabetes': diabetes,
            'heart_failure': heart_failure,
            'copd': copd,
            'hypertension': hypertension,
            'charlson_score': charlson_score,
            'prior_admissions_1yr': prior_admissions,
            'ed_visits_6m': ed_visits_6m,
            'num_medications': num_medications,
            'lives_alone': lives_alone,
            'transportation_issues': transportation_issues,
            'discharge_disposition': discharge_disposition,
            'readmitted_30d': readmitted_30d
        }
        
        patients.append(patient)
    
    return pd.DataFrame(patients)

# Generate and save the dataset
if __name__ == "__main__":
    df = generate_synthetic_patient_data(10000)
    df.to_csv('synthetic_patient_data.csv', index=False)
    print(f"Generated dataset with {len(df)} patients")
    print(f"Readmission rate: {df['readmitted_30d'].mean():.2%}")
    print("\nDataset saved as 'synthetic_patient_data.csv'")
