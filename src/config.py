import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATASET_PATH = os.path.join(DATA_DIR, 'raw', 'placement_data.csv')

# Output Directories
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'Plot')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'Report')
DBSCAN_DIR = os.path.join(OUTPUT_DIR, 'DBSCAN')

# Model Directory
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, 'placement_classifier.joblib')
REGRESSOR_MODEL_PATH = os.path.join(MODEL_DIR, 'salary_regressor.joblib')

# Flask App Configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5002
FLASK_DEBUG = True

# Machine Learning Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

CLEAN_DATA_PATH = os.path.join(DATA_DIR, 'clean', 'cleaned_placement_data.csv')
TARGET_COLUMN = 'PlacementStatus'
NUMERIC_COLUMNS = [
    'SGPA_Sem1', 'SGPA_Sem2', 'SGPA_Sem3', 'SGPA_Sem4', 
    'SGPA_Sem5', 'SGPA_Sem6', 'SGPA_Sem7', 'SGPA_Sem8', 
    'CGPA', 'AttendancePercent', 'Internships', 'Projects', 
    'Workshops', 'Certifications', 'Publications', 
    'AptitudeTestScore', 'SoftSkillsRating', 'CodingTestScore', 
    'MockInterviewScore'
]