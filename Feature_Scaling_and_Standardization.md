# Feature Scaling & Standardization

This document covers all the essential topics related to feature scaling and standardization, integrating concepts from your presentation into this machine learning project.

## Need for Feature Scaling
Machine learning algorithms often perform poorly if the features (variables) have different scales (e.g., one feature is measured in thousands, another in decimals). Feature scaling ensures that all features contribute equally to the model's training process, preventing features with larger magnitudes from dominating the objective function or distance calculations.

## Normalization
Normalization is the process of scaling the features of a dataset so that they fall within a specific range, usually between 0 and 1. It is useful when the data does not follow a Gaussian (normal) distribution and the algorithm you are using does not assume any distribution of the data (e.g., K-Nearest Neighbors and Artificial Neural Networks).

## Standardization
Standardization (or Z-score normalization) transforms the data such that the resulting distribution has a mean of 0 and a standard deviation of 1. It assumes that your data follows a Gaussian distribution. Even if the data isn't perfectly Gaussian, standardization often works well in practice and is less affected by outliers than normalization.

## Min-Max Normalization
Min-Max Normalization scales the values of a feature to a fixed range, typically [0, 1]. It preserves the exact shape of the original distribution but compresses it into the specified bounds.

## Min-Max Scaling Formula
The formula for Min-Max Scaling is:
$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
where:
- $X$ is the original value.
- $X_{min}$ is the minimum value in the feature column.
- $X_{max}$ is the maximum value in the feature column.

## Min-Max Scaling using Scikit-learn
In Python, this is implemented using `MinMaxScaler` from Scikit-learn:
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

## Outliers in Min-Max Scaling
Min-Max Scaling is highly sensitive to outliers. If a feature contains extreme outliers, $X_{min}$ or $X_{max}$ will be extreme, which compresses the vast majority of the data into a very narrow range (e.g., mostly between 0.0 and 0.1).

## When to Use Min-Max Scaling
Use Min-Max Scaling when:
- The upper and lower boundaries are well known from domain knowledge.
- The algorithm does not assume a specific distribution (e.g., Neural Networks, KNN).
- The data has no extreme outliers.
- You are working with image data (pixel intensities from 0 to 255).

## Z-Score Standardization
Z-Score Standardization centers the feature at mean 0 with a standard deviation of 1. It does not bound the data to a specific range, meaning extreme values are still preserved as extreme Z-scores.

**Formula:**
$$Z = \frac{X - \mu}{\sigma}$$
where:
- $\mu$ is the mean of the feature.
- $\sigma$ is the standard deviation of the feature.

## Standardization using Scikit-learn
In Python, this is implemented using `StandardScaler` from Scikit-learn:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Robust Scaling
Robust Scaling is similar to standardization but uses statistics that are robust to outliers. It removes the median and scales the data according to the Interquartile Range (IQR).

**Formula:**
$$X_{robust} = \frac{X - Q_2}{Q_3 - Q_1}$$
where $Q_2$ is the median, and $Q_1, Q_3$ are the 25th and 75th percentiles.

*Scikit-Learn implementation:* `RobustScaler`.

## Comparison of Min-Max, Standard, and Robust Scalers
| Feature | Min-Max Scaler | Standard Scaler | Robust Scaler |
| :--- | :--- | :--- | :--- |
| **Range** | [0, 1] | Unbounded (typically -3 to 3) | Unbounded |
| **Outlier Sensitivity** | Very High | Moderate | Low (Robust to outliers) |
| **Distribution Preserved?** | Yes | Yes | Yes |
| **Best For** | Images, Neural Nets, KNN | PCA, Linear/Logistic Regression | Datasets with many outliers |

## Impact of Scaling on KNN
K-Nearest Neighbors (KNN) relies entirely on distance metrics (like Euclidean distance). If one feature has a range of 1-1000 and another has a range of 0-1, the first feature will dominate the distance calculation. Scaling ensures all features contribute equally, drastically improving the accuracy of distance-based algorithms like KNN.

## Algorithms That Require Scaling
1. **Gradient Descent Based:** Linear Regression, Logistic Regression, Neural Networks.
2. **Distance-Based:** KNN, Support Vector Machines (SVM), K-Means Clustering.
3. **Dimensionality Reduction:** Principal Component Analysis (PCA).

## Algorithms That Do Not Require Scaling
Tree-based algorithms are scale-invariant. They work by partitioning the data at certain thresholds, which is unaffected by the scale of the variable.
- Decision Trees
- Random Forests (used in this project)
- Gradient Boosting (XGBoost, LightGBM, AdaBoost)
- Naive Bayes

*(Note: Although Random Forests are used in this project's `train_model.py`, scaling was included in the pipeline to ensure compatibility if other models like Logistic Regression or SVMs are tested later).*

## Train-Test Split and Scaling
You must **always split your data before scaling**. 
1. `fit()` the scaler on the **Training Data only**.
2. `transform()` the Training Data.
3. `transform()` the **Test Data** using the parameters (mean/variance/min/max) learned from the Training Data.

## Data Leakage in Scaling
Data leakage occurs when information from outside the training dataset is used to create the model. If you apply scaling to the entire dataset *before* performing the train-test split, the statistical properties of the test set (mean, max, min) "leak" into the training set, causing the model to artificially perform better during evaluation but fail in production.

## Common Scaling Mistakes
1. Scaling the entire dataset before splitting into train/test sets (Data Leakage).
2. Fitting the scaler on the test set (`scaler.fit(X_test)` instead of `scaler.transform(X_test)`).
3. Using Min-Max scaling on data heavily skewed by outliers.
4. Scaling target variables in classification problems (only independent features need scaling).

## Choosing the Appropriate Scaler
- **Use `StandardScaler`** as the default go-to scaler for most general-purpose machine learning tasks.
- **Use `MinMaxScaler`** when you need bounded values (e.g., image pixels, deep learning).
- **Use `RobustScaler`** when your data has extreme outliers that you cannot remove.

## Feature Scaling Best Practices
- Always split your data first.
- Use `Pipeline` from Scikit-Learn to prevent data leakage and bundle scaling with your model training.
- Save your fitted scaler (using `joblib` or `pickle`) along with your model, so you can scale new production data with the exact same parameters.

## Summary of Feature Scaling Techniques
Feature scaling is an essential data preprocessing step that standardizes the range of independent variables. Choosing the right scaling technique depends entirely on your dataset's distribution, the presence of outliers, and the specific machine learning algorithm you intend to use. Understanding and applying techniques like Min-Max, Standard, and Robust scaling appropriately will lead to faster convergence and better model performance.
