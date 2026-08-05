import os
import pandas as pd
from flask import Flask, render_template, send_from_directory, request, jsonify, make_response
import io

import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(BASE_DIR, "Output", "Plot")
REPORT_PATH = os.path.join(BASE_DIR, "Output", "Report")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load ML Models at startup
clf_pipeline = None
reg_pipeline = None
try:
    clf_pipeline = joblib.load(os.path.join(MODEL_DIR, "placement_classifier.joblib"))
    reg_pipeline = joblib.load(os.path.join(MODEL_DIR, "salary_regressor.joblib"))
    print("Machine Learning Models Loaded Successfully.")
except Exception as e:
    print("Warning: Could not load ML models:", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load')
def load_page():
    limit = request.args.get('limit', 10, type=int)
    csv_path = os.path.join(BASE_DIR, "Data", "placement_predict_50k Dataset (2).csv")
    try:
        df = pd.read_csv(csv_path)
        table_html = df.head(limit).to_html(classes="data-table", index=False)
        total_rows = len(df)
        total_cols = len(df.columns)
    except Exception as e:
        table_html = f"<p style='color: red;'>Error loading dataset: {e}</p>"
        total_rows = 0
        total_cols = 0
        
    return render_template('load.html', table_html=table_html, rows=total_rows, cols=total_cols, limit=limit)

@app.route('/eda')
def eda_page():
    try:
        plots = [f for f in os.listdir(PLOT_PATH) if f.endswith('.png')]
    except FileNotFoundError:
        plots = []
    
    try:
        with open(os.path.join(REPORT_PATH, "EDA Summary Report.txt"), "r") as f:
            report_text = f.read()
    except FileNotFoundError:
        report_text = "Report not found. Please run EDA script first."
        
    return render_template('eda.html', plots=plots, report=report_text)

@app.route('/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOT_PATH, filename)

@app.route('/feature_engg')
def feature_engg_page():
    return render_template('feature.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        data = request.json
        try:
            # Build input dataframe for the ML pipeline
            input_df = pd.DataFrame([{
                'CGPA': float(data.get('cgpa', 0)),
                'Internships': int(data.get('internships', 0)),
                'CodingTestScore': float(data.get('coding', 0)),
                'MockInterviewScore': float(data.get('mock', 0)),
                'AptitudeTestScore': float(data.get('aptitude', 0)),
                'SoftSkillsRating': float(data.get('soft_skills', 0)),
                'Projects': int(data.get('projects', 0)),
                'ExtraCurricular': int(data.get('extracurricular', 0)),
                'CollegeTier': data.get('tier', 'Tier2'),
                'Specialisation': data.get('specialisation', 'CS')
            }])
            
            if clf_pipeline and reg_pipeline:
                # Real ML Prediction
                pred_class = clf_pipeline.predict(input_df)[0]
                
                if pred_class == 1:
                    status = "Placed"
                    salary = round(reg_pipeline.predict(input_df)[0], 2)
                    message = "Strong candidate profile. High probability of placement according to the Random Forest Classification Model."
                    color = "#10b981" # success green
                    bg = "rgba(16, 185, 129, 0.1)"
                    border = "rgba(16, 185, 129, 0.3)"
                else:
                    status = "Not Placed"
                    salary = 0.0
                    message = "Model suggests lower probability of placement. Consider improving coding and mock interview skills."
                    color = "#ef4444" # danger red
                    bg = "rgba(239, 68, 68, 0.1)"
                    border = "rgba(239, 68, 68, 0.3)"
            else:
                return jsonify({'error': 'ML models are not loaded.'}), 500
                
            return jsonify({
                'status': status,
                'salary': f"{salary} LPA" if salary > 0 else "N/A",
                'message': message,
                'color': color,
                'bg': bg,
                'border': border
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    return render_template('predict.html')

@app.route('/evaluation')
def evaluation_page():
    return render_template('evaluation.html')

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('batch_predict.html', error='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('batch_predict.html', error='No selected file')
            
        if file and file.filename.endswith('.csv'):
            try:
                # Read uploaded CSV
                df = pd.read_csv(file)
                
                # Check if models are loaded
                if not clf_pipeline or not reg_pipeline:
                    return render_template('batch_predict.html', error='Machine Learning models are not loaded.')
                
                # Run predictions
                # Make sure the dataframe matches expected columns or handle gracefully
                # We assume the uploaded CSV has the correct feature names
                df['Predicted_Placement'] = clf_pipeline.predict(df)
                
                # Predict salary only for those predicted as placed (1)
                salaries = reg_pipeline.predict(df)
                df['Predicted_Salary_LPA'] = [round(sal, 2) if p == 1 else 0.0 for sal, p in zip(salaries, df['Predicted_Placement'])]
                
                # Map 1/0 to Placed/Not Placed
                df['Predicted_Placement'] = df['Predicted_Placement'].map({1: 'Placed', 0: 'Not Placed'})
                
                # Generate CSV to download
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                response = make_response(csv_buffer.getvalue())
                response.headers["Content-Disposition"] = "attachment; filename=Batch_Predictions_Result.csv"
                response.headers["Content-type"] = "text/csv"
                return response
            except Exception as e:
                return render_template('batch_predict.html', error=f"Error processing file: {str(e)}")
        else:
            return render_template('batch_predict.html', error='Invalid file format. Please upload a CSV.')
            
    return render_template('batch_predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
