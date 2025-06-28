# dropout_predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score,
    classification_report, confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")

# Step 1: Load Dataset
df = pd.read_csv('student_data.csv')
print("\n✅ Dataset Loaded:\n")
print(df.head())

# Step 2: Preprocessing
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df.drop('student_id', axis=1, inplace=True)

X = df.drop('dropout', axis=1)
y = df['dropout']

# Step 3: Split Dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.176, random_state=42
)

print(f"\n📊 Train Samples: {X_train.shape[0]}")
print(f"📊 Validation Samples: {X_val.shape[0]}")
print(f"📊 Test Samples: {X_test.shape[0]}")

# Step 4: Train Model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Step 5: Evaluate on Validation Set
y_val_pred = rf.predict(X_val)
print("\n📈 VALIDATION RESULTS")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Recall:", recall_score(y_val, y_val_pred))
print("\n", classification_report(y_val, y_val_pred))

# Step 6: Final Test Evaluation
y_test_pred = rf.predict(X_test)
print("\n✅ TEST RESULTS")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Step 7: Feature Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=rf.feature_importances_, y=X.columns)
plt.title("🎯 Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Step 8: Concept Drift Strategy
print("\n📌 CONCEPT DRIFT STRATEGY")
print("Concept drift occurs when student behavior changes over time.")
print("✅ Monitor accuracy and recall each semester.")
print("✅ Retrain model with fresh student data.")
print("✅ Use a rolling data window for updates.")
