import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.model_selection import train_test_split

def main():
    print("=== Categorical Encoding Experiments ===")
    
    # 1. Load Data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "cleaned_placement_data.csv")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # We will pick categorical features and the target variable to demonstrate
    cat_features = ["CollegeTier", "Specialisation"]
    target = "PlacementStatus"
    
    # Drop NaNs for demonstration purposes
    df = df.dropna(subset=cat_features + [target])
    
    X = df[cat_features]
    y = df[target]
    
    # --- 1. Original Data Statistics ---
    print("\n--- 1. Original Data Categories ---")
    for col in cat_features:
        print(f"{col}: {X[col].unique()}")
        
    # --- 2. Train-Test Split ---
    # BEST PRACTICE: Always split data before encoding to avoid Data Leakage (especially for TargetEncoder)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- 3. One-Hot Encoding ---
    print("\n--- 2. One-Hot Encoding (Nominal Data) ---")
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    # Fitting only on Specialisation to show effect
    X_train_onehot = onehot_encoder.fit_transform(X_train[["Specialisation"]])
    onehot_cols = onehot_encoder.get_feature_names_out(["Specialisation"])
    
    print(f"Original shape: {X_train[['Specialisation']].shape}")
    print(f"Encoded shape: {X_train_onehot.shape}")
    print(f"New Columns Created: {onehot_cols}")
    print("First 3 rows:\n", X_train_onehot[:3])
    
    # --- 4. Ordinal Encoding ---
    print("\n--- 3. Ordinal Encoding (Ordinal/Ranked Data) ---")
    # We explicitly define the order: Tier3 < Tier2 < Tier1
    ordinal_encoder = OrdinalEncoder(categories=[['Tier3', 'Tier2', 'Tier1']])
    X_train_ordinal = ordinal_encoder.fit_transform(X_train[["CollegeTier"]])
    
    print("Mapping: Tier3 -> 0, Tier2 -> 1, Tier1 -> 2")
    for i in range(3):
        print(f"Original: {X_train['CollegeTier'].iloc[i]} -> Encoded: {X_train_ordinal[i][0]}")
    
    # --- 5. Target Encoding ---
    print("\n--- 4. Target Encoding (High-Cardinality/Nominal Data) ---")
    # Replaces categories with the blended mean of the target variable
    target_encoder = TargetEncoder(smooth="auto")
    X_train_target = target_encoder.fit_transform(X_train[["Specialisation"]], y_train)
    
    print("Notice how each Specialisation is replaced by a probability-like float:")
    for i in range(5):
        print(f"Original: {X_train['Specialisation'].iloc[i]} -> Encoded: {X_train_target[i][0]:.4f}")
        
    print("\nExperiment Complete. The actual ML pipeline uses:")
    print("- OrdinalEncoder for CollegeTier")
    print("- OneHotEncoder for Specialisation")

if __name__ == "__main__":
    main()
