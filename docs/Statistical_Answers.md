# Statistical Analysis Report
This report addresses key questions regarding candidate placement and provides advanced statistical tests.

## Q1: Which factor influenced a candidate in getting placed?
To determine the factors, we analyzed correlations for numerical variables and performed Chi-Square tests for categorical variables against `PlacementStatus`.

### Numerical Factors (Point-Biserial Correlation)
| Feature | Correlation | P-Value | Significance |
|---------|-------------|---------|--------------|
| SGPA_Sem8 | 0.707 | 0.000e+00 | Significant |
| SGPA_Sem7 | 0.702 | 0.000e+00 | Significant |
| SGPA_Sem6 | 0.696 | 0.000e+00 | Significant |
| SGPA_Sem5 | 0.691 | 0.000e+00 | Significant |
| SGPA_Sem4 | 0.684 | 0.000e+00 | Significant |
| SGPA_Sem3 | 0.678 | 0.000e+00 | Significant |
| SGPA_Sem2 | 0.672 | 0.000e+00 | Significant |
| MockInterviewScore | 0.668 | 0.000e+00 | Significant |
| SGPA_Sem1 | 0.665 | 0.000e+00 | Significant |
| CGPA | 0.649 | 0.000e+00 | Significant |

**Insight**: CGPA and SGPA scores are highly correlated with placement status. `AptitudeTestScore` and `MockInterviewScore` also show significant positive influence.

### Categorical Factors (Chi-Square Test)
| Feature | Chi2 Statistic | P-Value | Significance |
|---------|----------------|---------|--------------|
| CollegeTier | 11169.03 | 0.000e+00 | Significant |
| HistoryOfBacklogs | 6220.19 | 0.000e+00 | Significant |
| CGPA_Tier | 24661.33 | 0.000e+00 | Significant |
| Stream | 19.28 | 1.704e-03 | Significant |
| Hostel | 0.81 | 3.670e-01 | Not Significant |
| Specialisation | 3.08 | 3.802e-01 | Not Significant |
| Gender | 0.47 | 4.943e-01 | Not Significant |
| City | 5.26 | 8.113e-01 | Not Significant |

**Insight**: `CollegeTier` and `Specialisation` have a highly significant relationship with placement status.

## Q2: Does percentage (CGPA) matter for one to get placed?
We conducted an independent two-sample T-test comparing the CGPA of placed versus unplaced candidates.

- **Mean CGPA (Placed)**: 8.00
- **Mean CGPA (Not Placed)**: 5.79
- **T-Statistic**: 200.34
- **P-Value**: 0.000e+00

**Conclusion**: Yes, percentage (CGPA) matters significantly. There is a statistically significant difference in CGPA between placed and non-placed candidates. Candidates with higher CGPAs have a much higher likelihood of placement.

## Q3: Which degree specialization is much demanded by corporate?
| Specialisation | Placement Rate (%) | Total Students |
|----------------|--------------------|----------------|
| AI | 66.23 | 11471 |
| Networking | 65.84 | 12904 |
| Embedded | 65.64 | 13614 |
| DataScience | 65.17 | 12011 |

**Insight**: Based on the placement rate, **AI** is the most demanded specialization by corporates.

## Q4: General Exploratory Statistical Tests
### 1. One-Way ANOVA: Salary Package across College Tiers
Are salaries significantly different between different college tiers?
- **F-Statistic**: 736.13
- **P-Value**: 1.811e-313
- **Conclusion**: There is a significant difference in salary packages across different college tiers.

### 2. Independent T-Test: Coding Test Score by Gender
- **Mean Score (Male)**: 57.55
- **Mean Score (Female)**: 57.21
- **P-Value**: 8.547e-02
- **Conclusion**: There is NO significant difference in Coding Test Scores between genders.

### 3. T-Test: Extra Curricular Activities vs Placement
- **P-Value**: 0.000e+00
- **Conclusion**: Extra-curricular activities significantly impact placement.
