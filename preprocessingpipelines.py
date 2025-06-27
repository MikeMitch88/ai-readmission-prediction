import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HospitalReadmissionPreprocessor:
    """
    Comprehensive preprocessing pipeline for hospital readmission prediction
    Handles EHR data cleaning, feature engineering, and transformation
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
    def load_and_validate_data(self, data_path=None, sample_data=None):
        """
        Load and perform initial data validation
        """
        if sample_data is not None:
            df = sample_data.copy()
        else:
            # In real implementation, load from EHR system
            df = self._generate_sample_data()
            
        print(f"Initial data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        return df
    
    def _generate_sample_data(self):
        """
        Generate sample hospital data for demonstration
        """
        np.random.seed(42)
        n_patients = 1000
        
        # Generate synthetic patient data
        data = {
            'patient_id': range(1, n_patients + 1),
            'age': np.random.normal(65, 15, n_patients).clip(18, 100),
            'gender': np.random.choice(['M', 'F'], n_patients),
            'insurance_type': np.random.choice(['Medicare', 'Medicaid', 'Private', 'Uninsured'], n_patients, p=[0.4, 0.2, 0.35, 0.05]),
            'admission_date': pd.date_range('2023-01-01', periods=n_patients, freq='D')[:n_patients],
            'discharge_date': None,  # Will be calculated
            'primary_diagnosis': np.random.choice(['Heart Disease', 'Diabetes', 'Pneumonia', 'COPD', 'Stroke'], n_patients),
            'num_comorbidities': np.random.poisson(2, n_patients),
            'num_medications': np.random.poisson(5, n_patients),
            'num_procedures': np.random.poisson(1, n_patients),
            'emergency_admission': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            'previous_admissions_30d': np.random.poisson(0.5, n_patients),
            'previous_admissions_1yr': np.random.poisson(2, n_patients),
            'systolic_bp_avg': np.random.normal(130, 20, n_patients),
            'diastolic_bp_avg': np.random.normal(80, 15, n_patients),
            'heart_rate_avg': np.random.normal(75, 15, n_patients),
            'temperature_max': np.random.normal(98.6, 1.5, n_patients),
            'lab_sodium': np.random.normal(140, 5, n_patients),
            'lab_potassium': np.random.normal(4.0, 0.5, n_patients),
            'lab_creatinine': np.random.gamma(2, 0.5, n_patients),
            'lab_hemoglobin': np.random.normal(12, 2, n_patients),
            'lab_white_blood_cell': np.random.gamma(3, 2, n_patients),
            'discharge_disposition': np.random.choice(['Home', 'SNF', 'Home_Health', 'AMA'], n_patients, p=[0.7, 0.15, 0.1, 0.05]),
            'social_support_score': np.random.randint(1, 6, n_patients),  # 1-5 scale
            'transportation_access': np.random.choice([0, 1], n_patients, p=[0.3, 0.7]),
            'housing_stability': np.random.choice([0, 1], n_patients, p=[0.2, 0.8]),
            'readmitted_30d': None  # Target variable - will be calculated
        }
        
        df = pd.DataFrame(data)
        
        # Calculate length of stay and discharge date
        los_days = np.random.gamma(2, 2, n_patients).clip(1, 30)
        df['discharge_date'] = df['admission_date'] + pd.to_timedelta(los_days, unit='D')
        df['length_of_stay'] = los_days
        
        # Introduce some missing values to simulate real data
        missing_cols = ['lab_sodium', 'lab_potassium', 'social_support_score', 'transportation_access']
        for col in missing_cols:
            missing_idx = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
            df.loc[missing_idx, col] = np.nan
            
        # Generate target variable (30-day readmission)
        # Higher risk factors increase readmission probability
        risk_score = (
            (df['age'] > 70).astype(int) * 0.3 +
            (df['num_comorbidities'] > 3).astype(int) * 0.4 +
            (df['emergency_admission'] == 1).astype(int) * 0.2 +
            (df['previous_admissions_30d'] > 0).astype(int) * 0.5 +
            (df['discharge_disposition'] != 'Home').astype(int) * 0.3 +
            (df['social_support_score'] < 3).astype(int) * 0.2
        )
        
        readmission_prob = 1 / (1 + np.exp(-2 * (risk_score - 1)))  # Sigmoid function
        df['readmitted_30d'] = np.random.binomial(1, readmission_prob)
        
        return df
    
    def clean_data(self, df):
        """
        Stage 1: Data cleaning and validation
        """
        print("=== STAGE 1: DATA CLEANING ===")
        
        # Remove duplicate patients
        initial_shape = df.shape
        df = df.drop_duplicates(subset=['patient_id'])
        print(f"Removed {initial_shape[0] - df.shape[0]} duplicate records")
        
        # Validate age ranges
        df = df[(df['age'] >= 0) & (df['age'] <= 120)]
        print(f"Removed records with invalid ages")
        
        # Clean vital signs (remove physiologically impossible values)
        df.loc[df['systolic_bp_avg'] < 60, 'systolic_bp_avg'] = np.nan
        df.loc[df['systolic_bp_avg'] > 250, 'systolic_bp_avg'] = np.nan
        df.loc[df['diastolic_bp_avg'] < 30, 'diastolic_bp_avg'] = np.nan
        df.loc[df['diastolic_bp_avg'] > 150, 'diastolic_bp_avg'] = np.nan
        df.loc[df['heart_rate_avg'] < 30, 'heart_rate_avg'] = np.nan
        df.loc[df['heart_rate_avg'] > 200, 'heart_rate_avg'] = np.nan
        
        # Clean lab values (remove extreme outliers)
        lab_ranges = {
            'lab_sodium': (120, 160),
            'lab_potassium': (2.0, 7.0),
            'lab_creatinine': (0.5, 15.0),
            'lab_hemoglobin': (5.0, 20.0),
            'lab_white_blood_cell': (1.0, 50.0)
        }
        
        for lab, (min_val, max_val) in lab_ranges.items():
            df.loc[(df[lab] < min_val) | (df[lab] > max_val), lab] = np.nan
            
        print(f"Cleaned data shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """
        Stage 2: Handle missing values with clinical knowledge
        """
        print("=== STAGE 2: MISSING VALUE IMPUTATION ===")
        
        # Define imputation strategies based on clinical knowledge
        
        # Simple imputation for categorical variables
        categorical_simple = ['insurance_type', 'discharge_disposition']
        for col in categorical_simple:
            if col in df.columns:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_value)
                print(f"Filled {col} missing values with mode: {mode_value}")
        
        # Median imputation for vital signs
        vital_signs = ['systolic_bp_avg', 'diastolic_bp_avg', 'heart_rate_avg', 'temperature_max']
        for col in vital_signs:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled {col} missing values with median: {median_val:.2f}")
        
        # KNN imputation for lab values (correlated with each other)
        lab_columns = ['lab_sodium', 'lab_potassium', 'lab_creatinine', 'lab_hemoglobin', 'lab_white_blood_cell']
        existing_lab_cols = [col for col in lab_columns if col in df.columns]
        
        if existing_lab_cols:
            knn_imputer = KNNImputer(n_neighbors=5)
            df[existing_lab_cols] = knn_imputer.fit_transform(df[existing_lab_cols])
            print(f"Applied KNN imputation to lab values: {existing_lab_cols}")
        
        # Forward fill for social determinants (assume stable over time)
        social_cols = ['social_support_score', 'transportation_access', 'housing_stability']
        for col in social_cols:
            if col in df.columns:
                # Fill with most common value if still missing
                if df[col].isnull().sum() > 0:
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    print(f"Filled {col} missing values with {fill_value}")
        
        return df
    
    def feature_engineering(self, df):
        """
        Stage 3: Feature engineering with clinical domain knowledge
        """
        print("=== STAGE 3: FEATURE ENGINEERING ===")
        
        # 1. Demographic Features
        df['age_group'] = pd.cut(df['age'], bins=[0, 50, 65, 80, 100], 
                                labels=['<50', '50-65', '65-80', '80+'])
        df['is_elderly'] = (df['age'] >= 65).astype(int)
        
        # 2. Comorbidity and Complexity Features
        df['high_comorbidity'] = (df['num_comorbidities'] >= 3).astype(int)
        df['polypharmacy'] = (df['num_medications'] >= 5).astype(int)
        df['medication_burden'] = df['num_medications'] / (df['age'] / 10)  # Normalized by age
        
        # 3. Clinical Severity Features
        df['abnormal_vitals'] = (
            ((df['systolic_bp_avg'] < 90) | (df['systolic_bp_avg'] > 180)) |
            ((df['diastolic_bp_avg'] < 60) | (df['diastolic_bp_avg'] > 110)) |
            ((df['heart_rate_avg'] < 60) | (df['heart_rate_avg'] > 100))
        ).astype(int)
        
        # Pulse pressure (cardiovascular risk indicator)
        df['pulse_pressure'] = df['systolic_bp_avg'] - df['diastolic_bp_avg']
        df['hypertensive'] = (df['systolic_bp_avg'] >= 140).astype(int)
        
        # 4. Laboratory-based Features
        # Anemia indicator
        df['anemic'] = ((df['gender'] == 'M') & (df['lab_hemoglobin'] < 13.5) |
                       (df['gender'] == 'F') & (df['lab_hemoglobin'] < 12.0)).astype(int)
        
        # Kidney function (estimated GFR)
        df['egfr'] = 175 * (df['lab_creatinine'] ** -1.154) * (df['age'] ** -0.203)
        df.loc[df['gender'] == 'F', 'egfr'] *= 0.742
        df['kidney_disease'] = (df['egfr'] < 60).astype(int)
        
        # Electrolyte imbalance
        df['electrolyte_imbalance'] = (
            (df['lab_sodium'] < 135) | (df['lab_sodium'] > 145) |
            (df['lab_potassium'] < 3.5) | (df['lab_potassium'] > 5.0)
        ).astype(int)
        
        # 5. Admission Pattern Features
        df['frequent_flyer'] = (df['previous_admissions_1yr'] >= 2).astype(int)
        df['recent_admission'] = (df['previous_admissions_30d'] > 0).astype(int)
        
        # Length of stay categories
        df['los_category'] = pd.cut(df['length_of_stay'], 
                                   bins=[0, 2, 5, 10, float('inf')],
                                   labels=['Short', 'Medium', 'Long', 'Extended'])
        df['prolonged_stay'] = (df['length_of_stay'] > 7).astype(int)
        
        # 6. Discharge and Social Features
        df['high_risk_discharge'] = (df['discharge_disposition'].isin(['SNF', 'AMA'])).astype(int)
        df['home_discharge'] = (df['discharge_disposition'] == 'Home').astype(int)
        
        # Social risk score
        df['social_risk_score'] = (
            (df['social_support_score'] <= 2).astype(int) * 2 +
            (df['transportation_access'] == 0).astype(int) * 1 +
            (df['housing_stability'] == 0).astype(int) * 2 +
            (df['insurance_type'] == 'Uninsured').astype(int) * 1
        )
        df['high_social_risk'] = (df['social_risk_score'] >= 3).astype(int)
        
        # 7. Diagnosis-specific Features
        # High-risk diagnoses for readmission
        high_risk_diagnoses = ['Heart Disease', 'COPD', 'Diabetes']
        df['high_risk_diagnosis'] = df['primary_diagnosis'].isin(high_risk_diagnoses).astype(int)
        
        # 8. Interaction Features
        df['elderly_emergency'] = df['is_elderly'] * df['emergency_admission']
        df['comorbid_elderly'] = df['high_comorbidity'] * df['is_elderly']
        df['social_medical_risk'] = df['high_social_risk'] * df['high_comorbidity']
        
        # 9. Temporal Features
        df['admission_day_of_week'] = df['admission_date'].dt.dayofweek
        df['weekend_admission'] = (df['admission_day_of_week'] >= 5).astype(int)
        df['admission_month'] = df['admission_date'].dt.month
        df['winter_admission'] = df['admission_month'].isin([12, 1, 2]).astype(int)
        
        print(f"Created {len([col for col in df.columns if col not in ['patient_id', 'admission_date', 'discharge_date', 'readmitted_30d']])} features")
        return df
    
    def encode_categorical_features(self, df):
        """
        Stage 4: Encode categorical variables
        """
        print("=== STAGE 4: CATEGORICAL ENCODING ===")
        
        # Define categorical columns
        categorical_cols = ['gender', 'insurance_type', 'primary_diagnosis', 
                          'discharge_disposition', 'age_group', 'los_category']
        
        # Label encoding for binary categories
        binary_cols = ['gender']
        for col in binary_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[col] = self.label_encoders[col].transform(df[col])
        
        # One-hot encoding for multi-class categories
        multi_class_cols = ['insurance_type', 'primary_diagnosis', 'discharge_disposition', 
                           'age_group', 'los_category']
        
        for col in multi_class_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
                print(f"One-hot encoded {col}: created {len(dummies.columns)} dummy variables")
        
        return df
    
    def normalize_features(self, df, fit=True):
        """
        Stage 5: Normalize numerical features
        """
        print("=== STAGE 5: FEATURE NORMALIZATION ===")
        
        # Identify numerical columns (exclude target and ID columns)
        exclude_cols = ['patient_id', 'readmitted_30d', 'admission_date', 'discharge_date']
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            print(f"Fitted scaler and normalized {len(numerical_cols)} numerical features")
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            print(f"Applied existing scaler to {len(numerical_cols)} numerical features")
        
        return df
    
    def fit_transform(self, df):
        """
        Complete preprocessing pipeline for training data
        """
        print("Starting preprocessing pipeline for training data...")
        
        # Execute all preprocessing stages
        df = self.clean_data(df)
        df = self.handle_missing_values(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical_features(df)
        df = self.normalize_features(df, fit=True)
        
        # Store feature names for later use
        exclude_cols = ['patient_id', 'readmitted_30d', 'admission_date', 'discharge_date']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        self.is_fitted = True
        
        print(f"Preprocessing complete. Final shape: {df.shape}")
        print(f"Features created: {len(self.feature_names)}")
        
        return df
    
    def transform(self, df):
        """
        Apply preprocessing to new data (validation/test)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming new data")
            
        print("Applying preprocessing pipeline to new data...")
        
        # Execute preprocessing stages (without fitting)
        df = self.clean_data(df)
        df = self.handle_missing_values(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical_features(df)
        df = self.normalize_features(df, fit=False)
        
        # Ensure same feature set as training
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            print(f"Warning: Missing features in new data: {missing_features}")
            for feature in missing_features:
                df[feature] = 0  # Add missing features with default value
        
        # Select only the features used in training
        df = df[['patient_id', 'readmitted_30d'] + self.feature_names]
        
        print(f"Transformation complete. Final shape: {df.shape}")
        return df
    
    def get_feature_info(self):
        """
        Return information about created features
        """
        if not self.is_fitted:
            return "Pipeline not fitted yet"
            
        feature_categories = {
            'Demographic': [f for f in self.feature_names if any(x in f.lower() for x in ['age', 'gender', 'elderly'])],
            'Clinical': [f for f in self.feature_names if any(x in f.lower() for x in ['vital', 'lab', 'bp', 'heart', 'temp', 'anemic', 'kidney'])],
            'Comorbidity': [f for f in self.feature_names if any(x in f.lower() for x in ['comorbid', 'medication', 'diagnosis'])],
            'Admission': [f for f in self.feature_names if any(x in f.lower() for x in ['admission', 'emergency', 'los', 'stay'])],
            'Social': [f for f in self.feature_names if any(x in f.lower() for x in ['social', 'support', 'transport', 'housing', 'insurance'])],
            'Discharge': [f for f in self.feature_names if any(x in f.lower() for x in ['discharge', 'home'])],
            'Interaction': [f for f in self.feature_names if any(x in f.lower() for x in ['elderly_', 'comorbid_', 'social_medical'])]
        }
        
        return feature_categories

# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = HospitalReadmissionPreprocessor()
    
    # Load sample data
    raw_data = preprocessor.load_and_validate_data()
    
    # Apply preprocessing pipeline
    processed_data = preprocessor.fit_transform(raw_data)
    
    # Display results
    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Original data shape: {raw_data.shape}")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Number of features created: {len(preprocessor.feature_names)}")
    
    # Show feature categories
    print("\n=== FEATURE CATEGORIES ===")
    feature_info = preprocessor.get_feature_info()
    for category, features in feature_info.items():
        print(f"{category}: {len(features)} features")
        if features:
            print(f"  Examples: {features[:3]}")
    
    # Show sample of processed data
    print("\n=== SAMPLE PROCESSED DATA ===")
    sample_features = preprocessor.feature_names[:10]
    print(processed_data[['patient_id', 'readmitted_30d'] + sample_features].head())
    
    # Show class distribution
    print(f"\n=== TARGET DISTRIBUTION ===")
    print(processed_data['readmitted_30d'].value_counts())
    print(f"Readmission rate: {processed_data['readmitted_30d'].mean():.2%}")
