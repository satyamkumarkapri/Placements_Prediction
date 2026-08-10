import pandas as pd
import numpy as np
import os

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the placement dataset from a CSV file.
    
    Args:
        filepath: Path to the CSV dataset.
        
    Returns:
        pd.DataFrame containing the dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    return pd.read_csv(filepath)

def get_feature_columns() -> list:
    """
    Returns the list of features used for training the Random Forest models.
    """
    return [
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

def get_numeric_features() -> list:
    """
    Returns the list of numeric features for the pipeline's ColumnTransformer.
    """
    return [
        "CGPA",
        "Internships",
        "CodingTestScore",
        "MockInterviewScore",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "Projects",
        "ExtraCurricular",
    ]

def get_categorical_features() -> list:
    """
    Returns the list of categorical features for the pipeline's ColumnTransformer.
    """
    return ["CollegeTier", "Specialisation"]

def format_prediction_input(
    cgpa: float,
    internships: int,
    coding: float,
    mock: float,
    aptitude: float,
    soft_skills: float,
    projects: int,
    extracurricular: int,
    tier: str,
    specialisation: str
) -> pd.DataFrame:
    """
    Formats individual raw inputs into a Pandas DataFrame compatible with the ML pipeline.
    
    Returns:
        pd.DataFrame with a single row containing the properly formatted features.
    """
    return pd.DataFrame([{
        'CGPA': float(cgpa),
        'Internships': int(internships),
        'CodingTestScore': float(coding),
        'MockInterviewScore': float(mock),
        'AptitudeTestScore': float(aptitude),
        'SoftSkillsRating': float(soft_skills),
        'Projects': int(projects),
        'ExtraCurricular': int(extracurricular),
        'CollegeTier': str(tier),
        'Specialisation': str(specialisation)
    }])

def clean_target_variable(df: pd.DataFrame, target_col: str = "PlacementStatus") -> pd.DataFrame:
    """
    Drops rows where the primary classification target variable is missing.
    """
    return df.dropna(subset=[target_col])

def clean_regression_target(df: pd.DataFrame, class_col: str = "PlacementStatus", reg_col: str = "Salary Package") -> pd.DataFrame:
    """
    Filters the dataset for only placed students and drops missing salary values.
    Useful for training the salary regressor model.
    """
    return df[df[class_col] == 1].dropna(subset=[reg_col])
