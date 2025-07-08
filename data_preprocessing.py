# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib

class ReadmissionPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        
    def fit_transform(self, df):
        """Fit preprocessor and transform data"""
        # Separate features and target
        X = df.drop(['patient_id', 'admission_date', 'discharge_date', 'readmitted_30d'], axis=1)
        y = df['readmitted_30d']
        
        # Identify categorical and numerical columns
        categorical_cols = ['gender', 'race', 'insurance', 'discharge_disposition']
        numerical_cols = [col for col in X.columns if col not in categorical_cols]
        
        # Process categorical variables
        X_categorical = X[categorical_cols]
        X_categorical_encoded = self.one_hot_encoder.fit_transform(X_categorical)
        
        # Get feature names for categorical variables
        categorical_feature_names = self.one_hot_encoder.get_feature_names_out(categorical_cols)
        
        # Process numerical variables
        X_numerical = X[numerical_cols]
        X_numerical_imputed = self.imputer.fit_transform(X_numerical)
        X_numerical_scaled = self.scaler.fit_transform(X_numerical_imputed)
        
        # Combine features
        X_processed = np.hstack([X_numerical_scaled, X_categorical_encoded])
        
        # Store feature names
        self.feature_names = list(numerical_cols) + list(categorical_feature_names)
        
        return X_processed, y
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        X = df.drop(['patient_id', 'admission_date', 'discharge_date', 'readmitted_30d'], axis=1, errors='ignore')
        
        categorical_cols = ['gender', 'race', 'insurance', 'discharge_disposition']
        numerical_cols = [col for col in X.columns if col not in categorical_cols]
        
        # Process categorical variables
        X_categorical = X[categorical_cols]
        X_categorical_encoded = self.one_hot_encoder.transform(X_categorical)
        
        # Process numerical variables
        X_numerical = X[numerical_cols]
        X_numerical_imputed = self.imputer.transform(X_numerical)
        X_numerical_scaled = self.scaler.transform(X_numerical_imputed)
        
        # Combine features
        X_processed = np.hstack([X_numerical_scaled, X_categorical_encoded])
        
        return X_processed
    
    def save(self, filepath):
        """Save preprocessor"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load preprocessor"""
        return joblib.load(filepath)

def preprocess_data():
    """Main preprocessing function"""
    print("Loading data...")
    df = pd.read_csv('synthetic_patient_data.csv')
    
    print("Preprocessing data...")
    preprocessor = ReadmissionPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Save preprocessor and data
    preprocessor.save('preprocessor.pkl')
    
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
    
    print(f"Data preprocessed and saved!")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Readmission rate in training: {y_train.mean():.2%}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

if __name__ == "__main__":
    preprocess_data()
