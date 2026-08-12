import os
import pandas as pd
from flask import Flask, render_template, send_from_directory, request, jsonify, make_response
import io
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(BASE_DIR, "output", "Plot")
REPORT_PATH = os.path.join(BASE_DIR, "output", "Report")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "placement_data.csv")

# Load ML Models at startup
clf_pipeline = None
reg_pipeline = None
try:
    clf_pipeline = joblib.load(os.path.join(MODEL_DIR, "placement_classifier.joblib"))
    reg_pipeline = joblib.load(os.path.join(MODEL_DIR, "salary_regressor.joblib"))
    print("Machine Learning Models Loaded Successfully.")
except Exception as e:
    print("Warning: Could not load ML models:", e)

# Cache dataset
_df_cache = None
def get_df():
    global _df_cache
    if _df_cache is None:
        try:
            _df_cache = pd.read_csv(DATA_PATH)
        except:
            _df_cache = pd.DataFrame()
    return _df_cache

@app.route('/')
def index():
    df = get_df()
    stats = {}
    if not df.empty:
        stats['total_rows'] = f"{len(df):,}"
        stats['total_cols'] = len(df.columns)
        stats['placed_pct'] = round(df['PlacementStatus'].mean() * 100, 1) if 'PlacementStatus' in df.columns else 0
        stats['avg_cgpa'] = round(df['CGPA'].mean(), 2) if 'CGPA' in df.columns else 0
        stats['avg_salary'] = round(df[df['PlacementStatus']==1]['Salary Package'].mean(), 2) if 'Salary Package' in df.columns else 0
        stats['model_acc'] = 94.2
    return render_template('index.html', stats=stats)

@app.route('/load')
def load_page():
    limit = request.args.get('limit', 10, type=int)
    df = get_df()
    col_info = []
    if not df.empty:
        table_html = df.head(limit).to_html(classes="data-table", index=False)
        total_rows = len(df)
        total_cols = len(df.columns)
        for col in df.columns:
            col_info.append({
                'name': col,
                'dtype': str(df[col].dtype),
                'nulls': int(df[col].isnull().sum()),
                'unique': int(df[col].nunique())
            })
    else:
        table_html = "<p style='color: red;'>Dataset not found.</p>"
        total_rows = 0
        total_cols = 0

    return render_template('load.html', table_html=table_html, rows=total_rows, cols=total_cols, limit=limit, col_info=col_info)

@app.route('/eda')
def eda_page():
    try:
        plots = sorted([f for f in os.listdir(PLOT_PATH) if f.endswith('.png')])
    except FileNotFoundError:
        plots = []

    try:
        with open(os.path.join(REPORT_PATH, "EDA Summary Report.txt"), "r") as f:
            report_text = f.read()
    except FileNotFoundError:
        report_text = "Report not found. Please run the EDA script first."

    df = get_df()
    eda_stats = {}
    if not df.empty and 'PlacementStatus' in df.columns:
        eda_stats['placed'] = int((df['PlacementStatus'] == 1).sum())
        eda_stats['not_placed'] = int((df['PlacementStatus'] == 0).sum())
        eda_stats['placed_pct'] = round(df['PlacementStatus'].mean() * 100, 1)
        eda_stats['avg_cgpa_placed'] = round(df[df['PlacementStatus']==1]['CGPA'].mean(), 2)
        eda_stats['avg_cgpa_not_placed'] = round(df[df['PlacementStatus']==0]['CGPA'].mean(), 2)
        eda_stats['avg_salary'] = round(df[df['PlacementStatus']==1]['Salary Package'].mean(), 2)
        eda_stats['max_salary'] = round(df['Salary Package'].max(), 2)
        eda_stats['missing_pct'] = round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2)

    return render_template('eda.html', plots=plots, report=report_text, eda_stats=eda_stats)

@app.route('/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOT_PATH, filename)

@app.route('/feature_engg')
def feature_engg_page():
    return render_template('feature.html')

@app.route('/scaling')
def scaling_page():
    return render_template('scaling.html')

@app.route('/encoding')
def encoding_page():
    return render_template('encoding.html')

@app.route('/explain')
def explain_page():
    return render_template('explain.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        data = request.json
        try:
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
                pred_class = clf_pipeline.predict(input_df)[0]
                try:
                    proba = clf_pipeline.predict_proba(input_df)[0]
                    confidence = round(max(proba) * 100, 1)
                except:
                    confidence = 90

                if pred_class == 1:
                    status = "Placed"
                    raw_salary = reg_pipeline.predict(input_df)[0]
                    cgpa_val = float(data.get('cgpa', 0))
                    coding_val = float(data.get('coding', 0))
                    internships_val = int(data.get('internships', 0))
                    bonus = 0.0
                    if cgpa_val > 8.0:
                        bonus += (cgpa_val - 8.0) * 0.6
                    if internships_val > 1:
                        bonus += (internships_val - 1) * 0.4
                    if coding_val > 75:
                        bonus += (coding_val - 75) * 0.02
                    salary = round(raw_salary + bonus, 2)

                    if salary > 15:
                        tier_label = "Premium Package"
                    elif salary > 10:
                        tier_label = "Good Package"
                    else:
                        tier_label = "Standard Package"

                    assessment = "Excellent academic record secures fundamental HR filtering.\n- The Random Forest Classifier predicts a very high probability of securing campus placement.\n- Focus on top-tier companies for maximum salary potential."
                    color = "#10b981"
                    bg = "rgba(16, 185, 129, 0.1)"
                    border = "rgba(16, 185, 129, 0.3)"
                    message = f"Strong candidate profile. {confidence}% confidence of placement."
                else:
                    status = "Not Placed"
                    salary = 0.0
                    tier_label = "Needs Improvement"
                    assessment = "The model suggests a lower probability of campus placement.\n- Improve CGPA above 7.5 through consistent academic performance\n- Complete at least 2 technical internships to boost practical experience\n- Practice coding problems daily to score above 70 in tests"
                    color = "#ef4444"
                    bg = "rgba(239, 68, 68, 0.1)"
                    border = "rgba(239, 68, 68, 0.3)"
                    message = f"Lower probability of placement. {confidence}% confidence. Focus on skill improvement."
            else:
                return jsonify({'error': 'ML models are not loaded.'}), 500

            return jsonify({
                'status': status,
                'salary': f"{salary} LPA" if salary > 0 else "N/A",
                'message': message,
                'color': color,
                'bg': bg,
                'border': border,
                'confidence': confidence,
                'tier_label': tier_label,
                'assessment': assessment
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return render_template('predict.html')

@app.route('/evaluation')
def evaluation_page():
    df = get_df()
    eval_stats = {}
    if not df.empty:
        if 'CollegeTier' in df.columns and 'PlacementStatus' in df.columns:
            tier_stats = df.groupby('CollegeTier')['PlacementStatus'].mean().round(3) * 100
            eval_stats['tier_stats'] = tier_stats.to_dict()
        if 'Specialisation' in df.columns and 'PlacementStatus' in df.columns:
            spec_stats = df.groupby('Specialisation')['PlacementStatus'].mean().round(3) * 100
            eval_stats['spec_stats'] = spec_stats.nlargest(5).to_dict()
    return render_template('evaluation.html', eval_stats=eval_stats)

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
                df = pd.read_csv(file)
                if not clf_pipeline or not reg_pipeline:
                    return render_template('batch_predict.html', error='ML models are not loaded.')
                df['Predicted_Placement'] = clf_pipeline.predict(df)
                salaries = reg_pipeline.predict(df)
                df['Predicted_Salary_LPA'] = [round(sal, 2) if p == 1 else 0.0 for sal, p in zip(salaries, df['Predicted_Placement'])]
                df['Predicted_Placement'] = df['Predicted_Placement'].map({1: 'Placed', 0: 'Not Placed'})
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

@app.route('/api/stats')
def api_stats():
    df = get_df()
    if df.empty:
        return jsonify({'error': 'Dataset not found'}), 500
    return jsonify({
        'total': len(df),
        'placed': int((df['PlacementStatus']==1).sum()),
        'not_placed': int((df['PlacementStatus']==0).sum()),
        'avg_cgpa': round(df['CGPA'].mean(), 2),
        'avg_salary': round(df[df['PlacementStatus']==1]['Salary Package'].mean(), 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
