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
# Count Plot
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="PlacementStatus")
plt.title("Placement Status Distribution")
save_plot("PlacementStatus.png")

# ==========================================
# Histogram
# ==========================================
plt.figure(figsize=(6,4))
plt.hist(df["CGPA"], bins=10, edgecolor="black")
plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Frequency")
save_plot("Histogram.png")

# ==========================================
# Pie Chart
# ==========================================
plt.figure(figsize=(6,6))
df["Gender"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.ylabel("")
plt.title("Gender Distribution")
save_plot("PieChart.png")

# ==========================================
# Scatter Plot
# ==========================================
plt.figure(figsize=(6,4))
sns.scatterplot(
    data=df,
    x="SGPA_Sem1",
    y="SGPA_Sem2"
)
plt.title("SGPA Sem1 vs SGPA Sem2")
save_plot("ScatterPlot.png")

# ==========================================
# Box Plot
# ==========================================
plt.figure(figsize=(6,4))
sns.boxplot(
    data=df,
    x="PlacementStatus",
    y="SGPA_Sem1"
)
plt.title("Placement Status vs SGPA")
save_plot("BoxPlot.png")

# ==========================================
# Count Plot
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(
    data=df,
    x="Gender",
    hue="PlacementStatus"
)
plt.title("Gender vs Placement")
save_plot("GenderPlacement.png")

# ==========================================
# Correlation Heatmap
# ==========================================
numeric = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,8))
sns.heatmap(
    numeric.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
save_plot("CorrelationHeatmap.png")

# ==========================================
# Report
# ==========================================
report = os.path.join(REPORT_PATH, "EDA Summary Report.txt")

with open(report, "w") as f:

    f.write("EDA SUMMARY REPORT\n")
    f.write("="*60 + "\n\n")

    f.write("Shape\n")
    f.write(str(df.shape))
    f.write("\n\n")

    f.write("Columns\n")
    f.write(str(df.columns.tolist()))
    f.write("\n\n")

    f.write("Missing Values\n")
    f.write(str(df.isnull().sum()))
    f.write("\n\n")

    f.write("Duplicate Rows\n")
    f.write(str(df.duplicated().sum()))
    f.write("\n\n")

    f.write("Statistics\n")
    f.write(str(df.describe(include="all")))

print("\n===================================")
print("EDA Completed Successfully")
print("===================================")
print("Plots Saved In:")
print(PLOT_PATH)
print("\nReport Saved In:")
print(report)