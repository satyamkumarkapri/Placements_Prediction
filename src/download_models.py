import os
import sys

if __name__ == "__main__":
    print("===============================================================")
    print("Intercepting legacy download_models.py script call...")
    print("Instead of downloading, we are natively training the models on Render")
    print("to guarantee scikit-learn version compatibility!")
    print("===============================================================")
    
    exit_code = os.system("python src/train_model.py")
    if exit_code != 0:
        print("Model training failed!", file=sys.stderr)
        sys.exit(1)
