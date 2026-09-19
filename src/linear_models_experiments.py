import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.linear_model import (
    LinearRegression,
    Ridge, Lasso, ElasticNet,
    RidgeCV, LassoCV, ElasticNetCV,
    LogisticRegression, LogisticRegressionCV
)
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "cleaned_placement_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_data():
    print(f"Loading dataset from: {DATA_PATH}\n")
    df = pd.read_csv(DATA_PATH)
    return df

def run_multilinear_regression(df):
    print("="*60)
    print("TOPIC 6: MULTILINEAR REGRESSION")
    print("="*60)
    
    features = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    data = df.dropna(subset=features + ["Salary Package"])
    X = data[features]
    y = data["Salary Package"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Intercept:", model.intercept_)
    for feature, coef in zip(features, model.coef_):
        print(f"{feature}: {coef:.4f}")

    print("\nModel Performance:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2:  {r2_score(y_test, y_pred):.4f}\n")


def run_logistic_scaler_comparison(df):
    print("="*60)
    print("TOPIC 7: LOGISTIC REGRESSION SCALER COMPARISON")
    print("="*60)
    
    features = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    data = df.dropna(subset=features + ["PlacementStatus"])
    X = data[features]
    y = data["PlacementStatus"]

    x_tr, x_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    def eval_model(x_train_sc, x_val_sc, label):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(x_train_sc, y_tr)
        acc = accuracy_score(y_val, model.predict(x_val_sc))
        print(f"{label:<20}: Validation Accuracy = {acc:.4f}")

    eval_model(x_tr, x_val, "Unscaled")
    
    std = StandardScaler()
    eval_model(std.fit_transform(x_tr), std.transform(x_val), "StandardScaler")
    
    mm = MinMaxScaler()
    eval_model(mm.fit_transform(x_tr), mm.transform(x_val), "MinMaxScaler\n")


def run_multinomial_logistic_regression(df):
    print("="*60)
    print("TOPIC 9: MULTINOMIAL LOGISTIC REGRESSION")
    print("="*60)

    features = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    data = df.dropna(subset=features + ["Salary Package"])
    X = data[features]
    salary_series = data["Salary Package"]
    salary_median = salary_series[salary_series > 0].median()

    def make_tier(salary):
        if salary == 0: return "Not Placed"
        elif salary < salary_median: return "Standard Package"
        return "Premium Package"

    y = salary_series.apply(make_tier)
    x_tr, x_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    x_tr_std = scaler.fit_transform(x_tr)
    x_val_std = scaler.transform(x_val)

    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(x_tr_std, y_tr)
    
    print(f"Validation Accuracy: {accuracy_score(y_val, model.predict(x_val_std)):.4f}\n")


def run_regularization_experiments(df):
    print("="*60)
    print("TOPIC 10: REGULARIZATION EXPERIMENTS (L1/L2/ElasticNet)")
    print("="*60)

    features = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "Internships", "Projects"]
    df = df.dropna(subset=features + ["PlacementStatus", "Salary Package"])
    X = df[features]
    
    # Classification
    y_class = df["PlacementStatus"]
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_c)
    X_te_s = scaler.transform(X_te_c)

    # Regression (Placed only)
    df_placed = df[df["PlacementStatus"] == 1]
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(df_placed[features], df_placed["Salary Package"], test_size=0.2, random_state=42)

    print("--- 1. Ridge (L2) Regression ---")
    pipe_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
    pipe_ridge.fit(X_tr_r, y_tr_r)
    print(f"Ridge Test R2: {r2_score(y_te_r, pipe_ridge.predict(X_te_r)):.4f}")

    print("\n--- 2. Lasso (L1) Regression with CV ---")
    pipe_lasso = Pipeline([('scaler', StandardScaler()), ('lasso', LassoCV(cv=5, max_iter=10000, random_state=42))])
    pipe_lasso.fit(X_tr_r, y_tr_r)
    print(f"Best alpha: {pipe_lasso['lasso'].alpha_:.4f}")
    print(f"Lasso Test R2: {r2_score(y_te_r, pipe_lasso.predict(X_te_r)):.4f}")
    
    # Save the Lasso model to disk (replacing train_regularization_models.py functionality)
    joblib.dump(pipe_lasso, os.path.join(MODELS_DIR, "lasso_regressor.joblib"))

    print("\n--- 3. Ridge Logistic Regression (L2) ---")
    lr_ridge = LogisticRegressionCV(Cs=10, penalty='l2', cv=5, solver='lbfgs', max_iter=500, random_state=42)
    lr_ridge.fit(X_tr_s, y_tr_c)
    print(f"Best C: {lr_ridge.C_[0]:.4f}")
    print(f"Accuracy: {accuracy_score(y_te_c, lr_ridge.predict(X_te_s)):.4f}")

    print("\n--- 4. Lasso Logistic Regression (L1) ---")
    lr_l1 = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=500, random_state=42)
    lr_l1.fit(X_tr_s, y_tr_c)
    print(f"Accuracy: {accuracy_score(y_te_c, lr_l1.predict(X_te_s)):.4f}")
    print(f"Zeros in coefficients: {np.sum(lr_l1.coef_ == 0)}\n")


def run_interpreting_model_coefficients(df):
    print("="*60)
    print("TOPIC 12: INTERPRETING MODEL COEFFICIENTS")
    print("="*60)

    features = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "Internships", "Projects"]
    data = df.dropna(subset=features + ["Salary Package"])
    X = data[features]
    y = data["Salary Package"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    print("Ridge Regression Coefficients (Standardized Features):")
    coef_dict = {feat: coef for feat, coef in zip(features, model.coef_)}
    sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for feat, coef in sorted_coefs:
        print(f"{feat:>20}: {coef:>8.4f}")
    
    print("\nInterpretation:")
    print("Because the features are standardized, the magnitude of the coefficient directly indicates feature importance.")
    print(f"The most important feature is {sorted_coefs[0][0]}.")
    print("="*60 + "\n")



if __name__ == "__main__":
    df = load_data()
    print("Note: Train/Val/Test Split (TOPIC 5) is demonstrated within each experiment.")
    run_multilinear_regression(df)
    run_logistic_scaler_comparison(df)
    run_multinomial_logistic_regression(df)
    run_regularization_experiments(df)
    run_interpreting_model_coefficients(df)
    print("Note: End-to-End Pipeline (TOPIC 11) is demonstrated in src/train_model.py")
    print("All linear model experiments completed successfully!")
