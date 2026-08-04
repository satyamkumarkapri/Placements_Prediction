import os
import urllib.request

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete!")
    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # URLs to the GitHub Release assets
    classifier_url = "https://github.com/satyamkumarkapri/Placements_Prediction/releases/download/v1.0/placement_classifier.joblib"
    regressor_url = "https://github.com/satyamkumarkapri/Placements_Prediction/releases/download/v1.0/salary_regressor.joblib"
    
    # Download them
    download_file(classifier_url, "models/placement_classifier.joblib")
    download_file(regressor_url, "models/salary_regressor.joblib")
