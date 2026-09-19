# 🎓 College Placement Prediction AI

![UI Preview](https://img.shields.io/badge/UI-Liquid_Glass_Morphism-4f46e5?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

An end-to-end Machine Learning pipeline and beautifully crafted interactive web dashboard designed to predict college student placements and estimate potential salary packages. 

This project goes beyond a simple ML script; it provides a **production-ready SaaS-like dashboard** complete with real-time inference monitoring, extensive Exploratory Data Analysis (EDA) visualizations, and batch processing capabilities.

## ✨ Key Features

### 🎨 Premium User Interface
- **Stunning Dark Mode UI**: A highly responsive, modern dashboard interface built from scratch using vanilla CSS glassmorphism, fluid animations, and a cohesive design system.
- **System Health Dashboard**: Real-time widgets monitor API latency, memory usage, and dataset connectivity, paired with a simulated live feed of recent prediction inferences.
- **Terminal Report Viewer**: A beautifully mocked macOS-style terminal window used to present the raw statistical EDA reports.

### 🧠 Advanced Machine Learning
- **Dual Predictive Models**: 
  - **Random Forest Classifier**: Predicts whether a student will be successfully placed (Placed vs. Not Placed) by analyzing their academic and extracurricular profile.
  - **Random Forest Regressor**: Estimates the expected salary package (in LPA) for successfully placed students.
- **Comprehensive Feature Set**: Model inference utilizes 10 critical datapoints including CGPA, Coding Test Scores, Mock Interview Scores, Aptitude, Soft Skills, and Extracurriculars.
- **Batch Processing Engine**: Upload CSV files containing hundreds of student profiles to run predictions in bulk, allowing universities to estimate placement statistics for entire cohorts simultaneously.

### 📊 Deep Data Analytics
- **Automated EDA**: Generates detailed Exploratory Data Analysis reports and high-resolution Univariate, Bivariate, and Multivariate visualizations dynamically.
- **Download Center**: A dedicated hub to view and export raw datasets, trained `.joblib` model files, and detailed statistical `.txt` reports for offline use.

## 📚 Applied Machine Learning Curriculum
This project serves as a comprehensive implementation of a 15-module curriculum, demonstrating practical application of both fundamental and advanced machine learning concepts:

### MODULE 1 • FOUNDATIONS
1. **Data Ingestion:** Automated loading of the 50k student record dataset (`src/clean_data.py`).
2. **Baseline Model and Serving:** End-to-end Flask integration serving baseline predictions (`app.py`).
3. **Exploratory Data Analysis (First Pass):** Interactive data profiling and distribution analysis (`templates/eda.html`).
4. **Data Cleaning:** Handling missing values, outlier removal, and dataset structuring (`src/clean_data.py`).

### MODULE 2 • LINEAR MODELS
5. **Train, Validation, and Test Splitting:** Strategic dataset partitioning for robust evaluation (`src/train_model.py`).
6. **Linear Regression Basics:** Implementation of multilinear regression for salary package estimation (`src/multilinear_regression.py`).
7. **Feature Scaling:** Comparative analysis of Min-Max, Standard, and Robust scalers (`templates/scaling.html`).
8. **Encoding and Imputation:** One-hot encoding for categoricals and strategy-based imputation (`templates/encoding.html`).
9. **Logistic Regression (Binary/Multi-Class):** Multinomial classification for placement status (`src/multinomial_logistic_regression.py`).
10. **Regularization:** L1 (Lasso) and L2 (Ridge) penalty experiments to prevent overfitting (`src/regularization_experiments.py`).
11. **Building an End-to-End Pipeline:** Scikit-Learn `ColumnTransformer` and `Pipeline` integration (`src/train_model.py`).
12. **Interpreting Model Coefficients:** Advanced SHAP (SHapley Additive exPlanations) value analysis for explainable AI (`templates/explain.html`).

### MODULE 3 • TREE-BASED MODELS
13. **Decision Trees:** Unconstrained and shallow tree structure analysis (`templates/decision_trees.html`).
14. **Pruning and Splitting Criteria:** Gini vs. Entropy evaluations and Cost Complexity Pruning (`ccp_alpha`).
15. **Bias-Variance Tradeoff via Tree Depth:** Visualizing the transition from high-bias underfitting to high-variance overfitting.

## 🛠️ Technology Stack

- **Backend / Web Framework**: Python 3, Flask, Jinja2
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Data Visualization**: Seaborn, Matplotlib
- **Frontend UI**: HTML5, Vanilla CSS3 (Custom Glassmorphism), JavaScript

## 🚀 Quick Start (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/satyamkumarkapri/Placements_Prediction.git
cd Placements_Prediction
```

### 2. Set up a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Models (Required)
*Note: Large model files are ignored by GitHub due to size constraints. You must generate them locally before running the app.*
```bash
python src/train_model.py
```
*This will process the 50k dataset, perform the necessary feature scaling, and generate the compiled `.joblib` files inside the `/models/` directory.*

### 5. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5001`.

## 📂 Project Architecture
```text
📦 Placements_Prediction
 ┣ 📂 data/               # CSV Datasets used for training
 ┃ ┣ 📂 raw/              # Raw data files (e.g. placement_data.csv)
 ┃ ┗ 📂 clean/            # Processed datasets
 ┣ 📂 output/             # Generated EDA Reports (TXT) and Visual Plots (PNG)
 ┣ 📂 docs/               # Documentation (e.g. Feature Scaling)
 ┣ 📂 src/                # Core ML Engine
 ┃ ┣ 📜 config.py         # Global configuration and path variables
 ┃ ┣ 📜 data_utils.py     # Data loading and preprocessing helpers
 ┃ ┣ 📜 train_model.py    # Pipeline to train Classifier and Regressor
 ┃ ┗ 📜 scaling_experiments.py # Tests for different feature scaling techniques
 ┣ 📂 models/             # Compiled .joblib models and fitted scalers
 ┣ 📂 static/             # Assets, Icons, and the custom style.css engine
 ┣ 📂 templates/          # HTML Jinja templates (Dashboard, EDA, Predict)
 ┣ 📜 app.py              # Main Flask server, API endpoints, and routing logic
 ┣ 📜 render.yaml         # Deployment configuration for Render.com
 ┗ 📜 requirements.txt    # Python dependencies
```

## 🤝 Contribution
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request if you have ideas for new features or UI improvements.
