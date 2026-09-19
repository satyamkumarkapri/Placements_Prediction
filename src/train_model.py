import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "cleaned_placement_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_IMG_DIR = os.path.join(BASE_DIR, "static", "images")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(STATIC_IMG_DIR):
    os.makedirs(STATIC_IMG_DIR)

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# We will use these features based on the form inputs
features = [
    "CGPA",
    "Internships",
    "CodingTestScore",
    "MockInterviewScore",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "Projects",
    "ExtraCurricular",
    "CollegeTier",
    "Specialisation",
]

# Ensure no NaNs in target
df = df.dropna(subset=["PlacementStatus"])

X = df[features]
y_class = df["PlacementStatus"]

# Prepare regression data (only for placed students)
df_placed = df[df["PlacementStatus"] == 1].dropna(subset=["Salary Package"])
X_reg = df_placed[features]
y_reg = df_placed["Salary Package"]

print("Building Preprocessing Pipelines...")
# Numeric and Categorical columns
numeric_features = [
    "CGPA",
    "Internships",
    "CodingTestScore",
    "MockInterviewScore",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "Projects",
    "ExtraCurricular",
]

# Preprocessing for numerical data
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Preprocessing for Specialisation (Nominal)
onehot_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Preprocessing for CollegeTier (Ordinal)
ordinal_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Tier3")), # default to lowest tier
        ("ordinal", OrdinalEncoder(categories=[['Tier3', 'Tier2', 'Tier1']], handle_unknown='use_encoded_value', unknown_value=-1)),
    ]
)

# Bundle preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("onehot", onehot_transformer, ["Specialisation"]),
        ("ordinal", ordinal_transformer, ["CollegeTier"]),
    ]
)

# Define the models
rf_clf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
xgb_clf = XGBClassifier(n_estimators=50, max_depth=10, random_state=42, use_label_encoder=False, eval_metric='logloss')
reg = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)

# Fit preprocessor first to transform data for SHAP
print("Training Classification Models...")
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

# Fit preprocessor
X_train_c_prep = preprocessor.fit_transform(X_train_c)
X_test_c_prep = preprocessor.transform(X_test_c)

# Train RF
rf_clf.fit(X_train_c_prep, y_train_c)
rf_acc = rf_clf.score(X_test_c_prep, y_test_c)
print(f"Random Forest Test Accuracy: {rf_acc:.3f}")

# Train XGBoost
xgb_clf.fit(X_train_c_prep, y_train_c)
xgb_acc = xgb_clf.score(X_test_c_prep, y_test_c)
print(f"XGBoost Test Accuracy: {xgb_acc:.3f}")

# Pick Best Classifier
best_clf = rf_clf if rf_acc >= xgb_acc else xgb_clf
best_clf_name = "RandomForest" if rf_acc >= xgb_acc else "XGBoost"
print(f"Best Model Selected: {best_clf_name} (Acc: {max(rf_acc, xgb_acc):.3f})")

# Bundle the best classifier in the pipeline
clf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", best_clf)])
reg_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", reg)])

# Linear pipelines
ridge_clf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42))])
lasso_clf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42))])
elasticnet_clf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, random_state=42))])

ridge_reg_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Ridge(alpha=1.0, random_state=42))])
lasso_reg_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Lasso(alpha=0.1, random_state=42))])
elasticnet_reg_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))])

print("Training Linear Classification Models...")
ridge_clf_pipeline.fit(X_train_c, y_train_c)
lasso_clf_pipeline.fit(X_train_c, y_train_c)
elasticnet_clf_pipeline.fit(X_train_c, y_train_c)

# Generate SHAP Plot
print("Generating SHAP Explainability Plot...")
X_sample = X_train_c_prep[:500] 
explainer = shap.TreeExplainer(best_clf)
shap_values = explainer.shap_values(X_sample)

# Get feature names from preprocessor
feature_names = numeric_features + \
                list(preprocessor.named_transformers_['onehot'].named_steps['onehot'].get_feature_names_out(["Specialisation"])) + \
                ["CollegeTier"]

# For binary classification, shap_values might be a list (for RF) or an array (for XGBoost)
if isinstance(shap_values, list):
    shap_vals_to_plot = shap_values[1]
else:
    shap_vals_to_plot = shap_values

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_to_plot, X_sample, feature_names=feature_names, show=False)
plt.title(f"SHAP Feature Importance ({best_clf_name})")
plt.tight_layout()
shap_path = os.path.join(STATIC_IMG_DIR, "shap_summary.png")
plt.savefig(shap_path)
plt.close()
print(f"SHAP plot saved to {shap_path}")

print("Training Regression Model...")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
reg_pipeline.fit(X_train_r, y_train_r)
print(f"Regression Test R2 Score: {reg_pipeline.score(X_test_r, y_test_r):.3f}")

print("Training Linear Regression Models...")
ridge_reg_pipeline.fit(X_train_r, y_train_r)
lasso_reg_pipeline.fit(X_train_r, y_train_r)
elasticnet_reg_pipeline.fit(X_train_r, y_train_r)

# Save models
print("Saving models to /models ...")
joblib.dump(clf_pipeline, os.path.join(MODEL_DIR, "placement_classifier.joblib"))
joblib.dump(reg_pipeline, os.path.join(MODEL_DIR, "salary_regressor.joblib"))

joblib.dump(ridge_clf_pipeline, os.path.join(MODEL_DIR, "ridge_classifier.joblib"))
joblib.dump(ridge_reg_pipeline, os.path.join(MODEL_DIR, "ridge_regressor.joblib"))

joblib.dump(lasso_clf_pipeline, os.path.join(MODEL_DIR, "lasso_classifier.joblib"))
joblib.dump(lasso_reg_pipeline, os.path.join(MODEL_DIR, "lasso_regressor.joblib"))

joblib.dump(elasticnet_clf_pipeline, os.path.join(MODEL_DIR, "elasticnet_classifier.joblib"))
joblib.dump(elasticnet_reg_pipeline, os.path.join(MODEL_DIR, "elasticnet_regressor.joblib"))

print("Training Complete! Models saved successfully.")
