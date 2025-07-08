# run_project.py
import sys
import os

def run_complete_project():
    """Run the complete readmission prediction project"""
    
    print("=== Hospital Readmission Prediction Project ===\n")
    
    # Step 1: Generate synthetic data
    print("Step 1: Generating synthetic patient data...")
    os.system('python synthetic_data_generator.py')
    
    # Step 2: Preprocess data  
    print("\nStep 2: Preprocessing data...")
    os.system('python data_preprocessing.py')
    
    # Step 3: Train models
    print("\nStep 3: Training models...")
    os.system('python model_training.py')
    
    # Step 4: Start API server
    print("\nStep 4: Starting API server...")
    print("API will be available at http://localhost:5000")
    print("Use Ctrl+C to stop the server")
    print("\nAPI Endpoints:")
    print("- GET /health - Health check")
    print("- GET /example_patient - Get example patient data")
    print("- POST /predict - Predict readmission risk")
    
    os.system('python api.py')

if __name__ == "__main__":
    run_complete_project()
