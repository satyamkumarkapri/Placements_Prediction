import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# Project Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "Data")
PLOT_PATH = os.path.join(BASE_DIR, "Output", "Plot")
REPORT_PATH = os.path.join(BASE_DIR, "Output", "Report")

os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

# ==========================================
# Show Available Files
# ==========================================
print("BASE_DIR :", BASE_DIR)
print("DATA_PATH:", DATA_PATH)

print("\nFiles inside Data Folder:")
print(os.listdir(DATA_PATH))

# ==========================================
# CSV Path
# ==========================================
CSV_PATH = os.path.join(DATA_PATH, "placement_predict_50k Dataset (2).csv")

print("\nCSV PATH:")
print(CSV_PATH)

print("\nFile Exists:", os.path.exists(CSV_PATH))

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv(CSV_PATH)

# ==========================================
# Dataset Information
# ==========================================
print("\nFirst 5 Rows")
print(df.head())

print("\nShape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nInfo")
df.info()

print("\nData Types")
print(df.dtypes)

print("\nStatistics")
print(df.describe(include="all"))

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

# ==========================================
# Save Plot Function
# ==========================================
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, filename), dpi=300)
    plt.close()

# ==========================================
# Placement Status
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="PlacementStatus", palette="viridis")
plt.title("Placement Status Distribution")
save_plot("PlacementStatus.png")

# ==========================================
# Univariate: Continuous
# ==========================================
plt.figure(figsize=(6,4))
sns.histplot(df["CGPA"], bins=15, kde=True, color="blue")
plt.title("CGPA Distribution")
save_plot("Histogram.png")

plt.figure(figsize=(6,4))
sns.histplot(df["MockInterviewScore"], bins=15, kde=True, color="green")
plt.title("Mock Interview Score Distribution")
save_plot("Uni_MockInterview.png")

plt.figure(figsize=(6,4))
sns.histplot(df["CodingTestScore"], bins=15, kde=True, color="purple")
plt.title("Coding Test Score Distribution")
save_plot("Uni_CodingTest.png")

# ==========================================
# Univariate: Categorical
# ==========================================
plt.figure(figsize=(6,6))
df["Gender"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, colors=["#66b3ff", "#ff9999"])
plt.ylabel("")
plt.title("Gender Distribution")
save_plot("PieChart.png")

plt.figure(figsize=(8,4))
sns.countplot(data=df, x="Specialisation", palette="Set2")
plt.title("Specialisation Distribution")
plt.xticks(rotation=45)
save_plot("Uni_Specialisation.png")

# ==========================================
# Bivariate: Continuous vs Target
# ==========================================
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="PlacementStatus", y="SGPA_Sem1", palette="Set1")
plt.title("Placement Status vs SGPA Sem1")
save_plot("BoxPlot.png")

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="PlacementStatus", y="CGPA", palette="Set1")
plt.title("Placement Status vs CGPA")
save_plot("Bi_CGPA_Placement.png")

# ==========================================
# Bivariate: Categorical vs Target
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Gender", hue="PlacementStatus", palette="Set2")
plt.title("Gender vs Placement")
save_plot("GenderPlacement.png")

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="CollegeTier", hue="PlacementStatus", palette="Set3")
plt.title("College Tier vs Placement")
save_plot("Bi_TierPlacement.png")

# ==========================================
# Scatter Plot
# ==========================================
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="SGPA_Sem1", y="SGPA_Sem2", alpha=0.5, color="teal")
plt.title("SGPA Sem1 vs SGPA Sem2")
save_plot("ScatterPlot.png")

# ==========================================
# Correlation Heatmap
# ==========================================
numeric = df.select_dtypes(include=np.number)
plt.figure(figsize=(10,8))
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
save_plot("CorrelationHeatmap.png")

# ==========================================
# Pairplot (Multivariate)
# ==========================================
cols_for_pairplot = ["CGPA", "CodingTestScore", "MockInterviewScore", "PlacementStatus"]
# Use a sample to avoid huge computation time if dataset is 50k
sample_df = df[cols_for_pairplot].sample(min(2000, len(df)), random_state=42)
sns.pairplot(sample_df, hue="PlacementStatus", palette="Set1", diag_kind="kde")
save_plot("Multi_PairPlot.png")

# ==========================================
# Report
# ==========================================
report = os.path.join(REPORT_PATH, "EDA Summary Report.txt")

import io

with open(report, "w") as f:
    # Header
    f.write("+" + "-"*78 + "+\n")
    f.write("|" + "ADVANCED EDA SUMMARY REPORT".center(78) + "|\n")
    f.write("+" + "-"*78 + "+\n\n")

    # 1. Dataset Overview
    f.write("[1] DATASET OVERVIEW\n")
    f.write("-" * 25 + "\n")
    f.write(f"> Total Rows       : {df.shape[0]:,}\n")
    f.write(f"> Total Features   : {df.shape[1]}\n")
    
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    memory_usage = [line for line in info_str.split('\\n') if 'memory usage' in line]
    if memory_usage:
        f.write(f"> {memory_usage[0].strip()}\n")
        
    num_cols = len(df.select_dtypes(include=np.number).columns)
    cat_cols = len(df.select_dtypes(exclude=np.number).columns)
    f.write(f"> Numeric Columns  : {num_cols}\n")
    f.write(f"> Categorical Cols : {cat_cols}\n\n")

    # 2. Missing Values Analysis
    f.write("[2] MISSING VALUES ANALYSIS\n")
    f.write("-" * 30 + "\n")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    if not missing_data.empty:
        for col, count in missing_data.items():
            pct = (count / len(df)) * 100
            f.write(f"  - {col:<20} : {count:>5} missing ({pct:>5.2f}%)\n")
    else:
        f.write("  > No missing values found in the dataset.\n")
    f.write("\n")

    # 3. Duplicate Analysis
    f.write("[3] DATA QUALITY\n")
    f.write("-" * 20 + "\n")
    duplicates = df.duplicated().sum()
    f.write(f"> Duplicate Rows   : {duplicates}\n")
    if duplicates > 0:
        f.write(f"> Duplicate Pct    : {(duplicates/len(df))*100:.2f}%\n")
    f.write("\n")

    # 4. Target Variable Distribution
    f.write("[4] TARGET VARIABLE (PlacementStatus)\n")
    f.write("-" * 40 + "\n")
    target_counts = df["PlacementStatus"].value_counts()
    target_pct = df["PlacementStatus"].value_counts(normalize=True) * 100
    for val in target_counts.index:
        status_label = "Placed" if val == 1 else "Not Placed"
        f.write(f"  - {status_label:<12} : {target_counts[val]:>6,} ({target_pct[val]:>5.2f}%)\n")
    f.write("\n")

    # 5. Key Statistical Insights
    f.write("[5] KEY NUMERICAL STATISTICS\n")
    f.write("-" * 30 + "\n")
    # Show describe for key columns only to keep it clean
    key_cols = ["CGPA", "CodingTestScore", "MockInterviewScore", "Salary Package"]
    if all(col in df.columns for col in key_cols):
        stats_df = df[key_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
        f.write(stats_df.to_string())
    else:
        f.write(str(df.describe()))
    f.write("\n\n")

    # 6. Correlation Highlights
    f.write("[6] CORRELATION HIGHLIGHTS (vs PlacementStatus)\n")
    f.write("-" * 50 + "\n")
    if "PlacementStatus" in numeric.columns:
        correlations = numeric.corr()["PlacementStatus"].drop("PlacementStatus").sort_values(ascending=False)
        f.write("Top 3 Positive Correlations:\n")
        for col, val in correlations.head(3).items():
            f.write(f"  + {col:<20} : {val:.3f}\n")
        
        f.write("\nTop 3 Negative/Weakest Correlations:\n")
        for col, val in correlations.tail(3).items():
            f.write(f"  - {col:<20} : {val:.3f}\n")
    f.write("\n")
    
    f.write("+" + "-"*78 + "+\n")
    f.write("|" + "END OF REPORT".center(78) + "|\n")
    f.write("+" + "-"*78 + "+\n")


print("EDA Completed Successfully")
print("===================================")
print("Plots Saved In:")
print(PLOT_PATH)
print("\nReport Saved In:")
print(report)