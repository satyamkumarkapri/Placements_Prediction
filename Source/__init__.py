"""
Source Module for Placement Analytics

This package contains the Machine Learning training scripts, data utility
functions, and Exploratory Data Analysis (EDA) pipelines for the Placement 
Predictor system.
"""

from .data_utils import (
    load_dataset,
    get_feature_columns,
    get_numeric_features,
    get_categorical_features,
    format_prediction_input,
    clean_target_variable,
    clean_regression_target
)

__all__ = [
    "load_dataset",
    "get_feature_columns",
    "get_numeric_features",
    "get_categorical_features",
    "format_prediction_input",
    "clean_target_variable",
    "clean_regression_target"
]
