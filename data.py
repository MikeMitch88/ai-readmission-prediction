# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

# Step 2: Load the data
try:
    df = pd.read_csv('diabetic_data.csv')
except FileNotFoundError:
    print("diabetic_data.csv not found. Please upload the file.")
    uploaded = files.upload()
    for fn in uploaded.keys():
        print('User uploaded file "{name}" with length {length} bytes'.format(
            name=fn, length=len(uploaded[fn])))
        df = pd.read_csv(fn)


# Step 3: Handle missing values in 'race' and drop rows with unknown gender
df = df[df['gender'] != 'Unknown/Invalid']
df['race'].fillna('Unknown', inplace=True)

# Step 4: Convert readmitted column to binary target variable
# <30 → 1 (readmitted within 30 days), others → 0
df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Step 5: Select relevant features (you can expand this list as needed)
selected_features = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses'
]

# Step 6: Prepare input and target
X = df[selected_features]
y = df['readmitted_binary']

# Step 7: Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Precision and Recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")