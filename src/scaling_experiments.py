import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== Feature Scaling & Standardization Experiments ===")
    
    # 1. Load Data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_50k Dataset (2).csv")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # We will pick a few numerical features to demonstrate scaling
    features = ["CGPA", "CodingTestScore", "Salary Package"]
    
    # Drop NaNs for demonstration purposes
    df = df.dropna(subset=features)
    
    X = df[features]
    
    print("\n--- 1. Original Data Statistics ---")
    print(X.describe().loc[['mean', 'std', 'min', 'max']])
    
    # --- 2. Train-Test Split ---
    # BEST PRACTICE: Always split data before scaling to avoid Data Leakage
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    print("\n--- 2. Min-Max Scaling (Normalization) ---")
    # Bounds data between 0 and 1
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_test_minmax = min_max_scaler.transform(X_test)
    
    df_minmax = pd.DataFrame(X_train_minmax, columns=features)
    print("Min-Max Scaled Data Statistics (Training Set):")
    print(df_minmax.describe().loc[['mean', 'std', 'min', 'max']])
    
    print("\n--- 3. Z-Score Standardization ---")
    # Mean = 0, Standard Deviation = 1
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    
    df_std = pd.DataFrame(X_train_std, columns=features)
    print("Standardized Data Statistics (Training Set):")
    print(df_std.describe().loc[['mean', 'std', 'min', 'max']])
    
    print("\n--- 4. Robust Scaling ---")
    # Robust to outliers (uses Median and IQR)
    robust_scaler = RobustScaler()
    X_train_robust = robust_scaler.fit_transform(X_train)
    X_test_robust = robust_scaler.transform(X_test)
    
    df_robust = pd.DataFrame(X_train_robust, columns=features)
    print("Robust Scaled Data Statistics (Training Set):")
    print(df_robust.describe().loc[['mean', 'std', 'min', 'max']])
    
    print("\n--- 5. Simulating Data Leakage Mistake ---")
    # MISTAKE: Scaling the entire dataset before splitting
    leaky_scaler = StandardScaler()
    X_leaky = leaky_scaler.fit_transform(X) # Fitting on everything!
    X_train_leak, X_test_leak = train_test_split(X_leaky, test_size=0.2, random_state=42)
    
    print("Notice how the test set parameters leaked into the scaler's knowledge.")
    print(f"Proper Scaler Mean learned (Train only): {std_scaler.mean_}")
    print(f"Leaky Scaler Mean learned (All data):    {leaky_scaler.mean_}")
    
    print("\nExperiment Complete. Review the code to see practical implementations of:")
    print("- Min-Max Normalization")
    print("- Z-Score Standardization")
    print("- Robust Scaling")
    print("- Train-Test Split best practices")

if __name__ == "__main__":
    main()
