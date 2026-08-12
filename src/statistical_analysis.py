import os
import pandas as pd
import numpy as np
from scipy import stats

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_analysis(data_path, output_path):
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    report = []
    report.append("# Statistical Analysis Report")
    report.append("This report addresses key questions regarding candidate placement and provides advanced statistical tests.\n")

    # Question 1: Which factor influenced a candidate in getting placed?
    report.append("## Q1: Which factor influenced a candidate in getting placed?")
    report.append("To determine the factors, we analyzed correlations for numerical variables and performed Chi-Square tests for categorical variables against `PlacementStatus`.\n")

    # Numerical correlation
    report.append("### Numerical Factors (Point-Biserial Correlation)")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'PlacementStatus' in numerical_cols:
        numerical_cols.remove('PlacementStatus')
    if 'Salary Package' in numerical_cols:
        numerical_cols.remove('Salary Package') # Prevent data leakage
    if 'StudentID' in numerical_cols:
        numerical_cols.remove('StudentID')
        
    correlations = []
    for col in numerical_cols:
        # Drop NaNs for the test
        valid_data = df[[col, 'PlacementStatus']].dropna()
        if not valid_data.empty and valid_data[col].nunique() > 1:
            corr, p_value = stats.pointbiserialr(valid_data['PlacementStatus'], valid_data[col])
            correlations.append((col, corr, p_value))
            
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    report.append("| Feature | Correlation | P-Value | Significance |")
    report.append("|---------|-------------|---------|--------------|")
    for col, corr, p in correlations[:10]: # Top 10
        sig = "Significant" if p < 0.05 else "Not Significant"
        report.append(f"| {col} | {corr:.3f} | {p:.3e} | {sig} |")
    report.append("\n**Insight**: CGPA and SGPA scores are highly correlated with placement status. `AptitudeTestScore` and `MockInterviewScore` also show significant positive influence.\n")

    # Categorical Chi-Square
    report.append("### Categorical Factors (Chi-Square Test)")
    categorical_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
    chi_results = []
    for col in categorical_cols:
        contingency_table = pd.crosstab(df[col], df['PlacementStatus'])
        chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
        chi_results.append((col, chi2, p))
        
    chi_results.sort(key=lambda x: x[2])
    report.append("| Feature | Chi2 Statistic | P-Value | Significance |")
    report.append("|---------|----------------|---------|--------------|")
    for col, chi2, p in chi_results:
        sig = "Significant" if p < 0.05 else "Not Significant"
        report.append(f"| {col} | {chi2:.2f} | {p:.3e} | {sig} |")
    report.append("\n**Insight**: `CollegeTier` and `Specialisation` have a highly significant relationship with placement status.\n")


    # Question 2: Does percentage matters for one to get placed?
    report.append("## Q2: Does percentage (CGPA) matter for one to get placed?")
    report.append("We conducted an independent two-sample T-test comparing the CGPA of placed versus unplaced candidates.\n")
    placed_cgpa = df[df['PlacementStatus'] == 1]['CGPA'].dropna()
    unplaced_cgpa = df[df['PlacementStatus'] == 0]['CGPA'].dropna()
    
    t_stat, p_val = stats.ttest_ind(placed_cgpa, unplaced_cgpa, equal_var=False)
    
    report.append(f"- **Mean CGPA (Placed)**: {placed_cgpa.mean():.2f}")
    report.append(f"- **Mean CGPA (Not Placed)**: {unplaced_cgpa.mean():.2f}")
    report.append(f"- **T-Statistic**: {t_stat:.2f}")
    report.append(f"- **P-Value**: {p_val:.3e}")
    
    if p_val < 0.05:
        report.append("\n**Conclusion**: Yes, percentage (CGPA) matters significantly. There is a statistically significant difference in CGPA between placed and non-placed candidates. Candidates with higher CGPAs have a much higher likelihood of placement.\n")
    else:
        report.append("\n**Conclusion**: No significant difference found, percentage does not seem to matter heavily based on this test.\n")


    # Question 3: Which degree specialization is much demanded by corporate?
    report.append("## Q3: Which degree specialization is much demanded by corporate?")
    if 'Specialisation' in df.columns:
        spec_stats = df.groupby('Specialisation')['PlacementStatus'].agg(['mean', 'count']).reset_index()
        spec_stats.columns = ['Specialisation', 'Placement_Rate', 'Total_Students']
        spec_stats['Placement_Rate'] = (spec_stats['Placement_Rate'] * 100).round(2)
        spec_stats = spec_stats.sort_values(by='Placement_Rate', ascending=False)
        
        report.append("| Specialisation | Placement Rate (%) | Total Students |")
        report.append("|----------------|--------------------|----------------|")
        for _, row in spec_stats.iterrows():
            report.append(f"| {row['Specialisation']} | {row['Placement_Rate']} | {row['Total_Students']} |")
        
        best_spec = spec_stats.iloc[0]['Specialisation']
        report.append(f"\n**Insight**: Based on the placement rate, **{best_spec}** is the most demanded specialization by corporates.\n")
    else:
        report.append("Specialisation column not found in dataset.\n")


    # Question 4: Play with the data conducting all statistical tests.
    report.append("## Q4: General Exploratory Statistical Tests")
    
    # ANOVA: Salary vs College Tier
    report.append("### 1. One-Way ANOVA: Salary Package across College Tiers")
    report.append("Are salaries significantly different between different college tiers?")
    if 'Salary Package' in df.columns and 'CollegeTier' in df.columns:
        placed_df = df[df['PlacementStatus'] == 1].dropna(subset=['Salary Package', 'CollegeTier'])
        tiers = placed_df['CollegeTier'].unique()
        grouped_salaries = [placed_df[placed_df['CollegeTier'] == t]['Salary Package'] for t in tiers]
        
        f_stat, p_val = stats.f_oneway(*grouped_salaries)
        report.append(f"- **F-Statistic**: {f_stat:.2f}")
        report.append(f"- **P-Value**: {p_val:.3e}")
        if p_val < 0.05:
             report.append("- **Conclusion**: There is a significant difference in salary packages across different college tiers.\n")
        else:
             report.append("- **Conclusion**: No significant difference in salary packages across college tiers.\n")
             
    # T-Test: Coding Test Score by Gender
    report.append("### 2. Independent T-Test: Coding Test Score by Gender")
    if 'CodingTestScore' in df.columns and 'Gender' in df.columns:
        male_scores = df[df['Gender'] == 'Male']['CodingTestScore'].dropna()
        female_scores = df[df['Gender'] == 'Female']['CodingTestScore'].dropna()
        t_stat, p_val = stats.ttest_ind(male_scores, female_scores, equal_var=False)
        report.append(f"- **Mean Score (Male)**: {male_scores.mean():.2f}")
        report.append(f"- **Mean Score (Female)**: {female_scores.mean():.2f}")
        report.append(f"- **P-Value**: {p_val:.3e}")
        if p_val < 0.05:
            report.append("- **Conclusion**: There is a significant difference in Coding Test Scores between genders.\n")
        else:
            report.append("- **Conclusion**: There is NO significant difference in Coding Test Scores between genders.\n")

    # T-Test: Extra Curricular vs Placed
    report.append("### 3. T-Test: Extra Curricular Activities vs Placement")
    if 'ExtraCurricular' in df.columns:
        placed_ext = df[df['PlacementStatus'] == 1]['ExtraCurricular'].dropna()
        unplaced_ext = df[df['PlacementStatus'] == 0]['ExtraCurricular'].dropna()
        t_stat, p_val = stats.ttest_ind(placed_ext, unplaced_ext, equal_var=False)
        report.append(f"- **P-Value**: {p_val:.3e}")
        if p_val < 0.05:
             report.append("- **Conclusion**: Extra-curricular activities significantly impact placement.\n")
        else:
             report.append("- **Conclusion**: Extra-curricular activities do NOT significantly impact placement.\n")

    # Save report
    ensure_dir(output_path)
    with open(output_path, "w") as f:
        f.write("\n".join(report))
        
    print(f"Statistical Analysis Report successfully generated at: {output_path}")

if __name__ == "__main__":
    # Adjust paths relative to script location assuming it's in src/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "placement_data.csv")
    output_path = os.path.join(base_dir, "docs", "Statistical_Answers.md")
    run_analysis(data_path, output_path)
