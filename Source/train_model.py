import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_50k Dataset (2).csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# We will use these features based on the form inputs
features = [
    'CGPA', 'Internships', 'CodingTestScore', 'MockInterviewScore', 
    'AptitudeTestScore', 'SoftSkillsRating', 'Projects', 'ExtraCurricular',
    'CollegeTier', 'Specialisation'
]

# Ensure no NaNs in target
df = df.dropna(subset=['PlacementStatus'])

X = df[features]
y_class = df['PlacementStatus']

# Prepare regression data (only for placed students)
df_placed = df[df['PlacementStatus'] == 1].dropna(subset=['Salary Package'])
X_reg = df_placed[features]
y_reg = df_placed['Salary Package']

print("Building Preprocessing Pipelines...")
# Numeric and Categorical columns
numeric_features = [
    'CGPA', 'Internships', 'CodingTestScore', 'MockInterviewScore',
    'AptitudeTestScore', 'SoftSkillsRating', 'Projects', 'ExtraCurricular'
]
categorical_features = ['CollegeTier', 'Specialisation']

# Preprocessing for numerical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the models
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Bundle preprocessing and modeling code in a pipeline
clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', reg)])

print("Training Classification Model...")
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
clf_pipeline.fit(X_train_c, y_train_c)
print(f"Classification Train Accuracy: {clf_pipeline.score(X_train_c, y_train_c):.3f}")
print(f"Classification Test Accuracy: {clf_pipeline.score(X_test_c, y_test_c):.3f}")

print("Training Regression Model...")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_pipeline.fit(X_train_r, y_train_r)
print(f"Regression Test R2 Score: {reg_pipeline.score(X_test_r, y_test_r):.3f}")

# Save models
print("Saving models to /models ...")
joblib.dump(clf_pipeline, os.path.join(MODEL_DIR, "placement_classifier.joblib"))
joblib.dump(reg_pipeline, os.path.join(MODEL_DIR, "salary_regressor.joblib"))

print("Training Complete! Models saved successfully.")
