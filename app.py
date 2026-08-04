import os
import pandas as pd
from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(BASE_DIR, "Output", "Plot")
REPORT_PATH = os.path.join(BASE_DIR, "Output", "Report")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load')
def load_page():
    csv_path = os.path.join(BASE_DIR, "Data", "placement_predict_50k Dataset (2).csv")
    try:
        df = pd.read_csv(csv_path)
        table_html = df.head(10).to_html(classes="data-table", index=False)
        total_rows = len(df)
        total_cols = len(df.columns)
    except Exception as e:
        table_html = f"<p style='color: red;'>Error loading dataset: {e}</p>"
        total_rows = 0
        total_cols = 0
        
    return render_template('load.html', table_html=table_html, rows=total_rows, cols=total_cols)

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

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
