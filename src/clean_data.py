import os
import pandas as pd

def main():
    print("Starting Data Cleaning Process...")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "placement_data.csv")
    CLEAN_DATA_DIR = os.path.join(BASE_DIR, "data", "clean")
    CLEAN_DATA_PATH = os.path.join(CLEAN_DATA_DIR, "cleaned_placement_data.csv")
    
    if not os.path.exists(CLEAN_DATA_DIR):
        os.makedirs(CLEAN_DATA_DIR)
        
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw dataset not found at {RAW_DATA_PATH}")
        return
        
    # 1. Load Raw Data
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw data loaded. Shape: {df.shape}")
    
    # 2. Handle Missing Values
    # Drop rows where the target variable is missing
    df_clean = df.dropna(subset=["PlacementStatus"]).copy()
    
    # Fill or drop other critical missing values if necessary
    # (The pipeline's SimpleImputer handles most, but we can do a baseline drop here for severe missingness)
    df_clean = df_clean.dropna(subset=["CGPA", "Specialisation", "CollegeTier"])
    
    print(f"Cleaned data shape after dropping critical NaNs: {df_clean.shape}")
    
    # 3. Save Cleaned Data
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Clean data successfully saved to: {CLEAN_DATA_PATH}")

if __name__ == "__main__":
    main()
