# 🎓 College Placement Prediction AI

![UI Preview](https://img.shields.io/badge/UI-Liquid_Glass_Morphism-4f46e5?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

An end-to-end Machine Learning pipeline and beautifully crafted interactive web dashboard designed to predict college student placements and estimate potential salary packages. 

## ✨ Key Features

- **Stunning Liquid Glass UI**: A highly responsive, modern dashboard interface built from scratch using vanilla CSS glassmorphism, fluid animations, and a cohesive design system.
- **Dual Predictive Models**: 
  - **Random Forest Classifier**: Predicts whether a student will be placed (Placed vs. Not Placed).
  - **Random Forest Regressor**: Estimates the expected salary package for successfully placed students.
- **Comprehensive Feature Set**: Model inference utilizes 10 critical datapoints including CGPA, Coding Test Scores, Mock Interview Scores, Aptitude, Soft Skills, and Extracurriculars.
- **Automated EDA**: Generates detailed Exploratory Data Analysis reports and high-resolution Univariate, Bivariate, and Multivariate visualizations dynamically.

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
python Source/train_model.py
```
*This will process the 50k dataset and generate the necessary `.joblib` files inside the `/models/` directory.*

### 5. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5001`.

## 📂 Project Structure
```text
📦 Placements_Prediction
 ┣ 📂 Data/               # Raw 50k CSV Datasets
 ┣ 📂 Output/             # Generated EDA Reports and PNG Plots
 ┣ 📂 Source/             # ML Training and EDA Scripts
 ┣ 📂 models/             # Compiled .joblib models (Generated locally)
 ┣ 📂 static/             # CSS stylesheets and assets
 ┣ 📂 templates/          # HTML Jinja templates (Dashboard, EDA, Predict)
 ┣ 📜 app.py              # Main Flask server and API routes
 ┗ 📜 requirements.txt    # Python dependencies
```

## 🤝 Contribution
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
