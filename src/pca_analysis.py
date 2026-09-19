import sys, os

# Add the project root directory to Python's search path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

import numpy as np
import pandas as pd
import matplotlib

# Use a non-GUI backend for saving plots
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import config


# Create folder to store PCA results
PCA_DIR = os.path.join(config.OUTPUT_DIR, "PCA")
os.makedirs(PCA_DIR, exist_ok=True)


# ---------- Load and preprocess data ----------

# Read the cleaned student dataset
df = pd.read_csv(config.CLEAN_DATA_PATH)

# Select only the numeric features used for PCA
feature_cols = config.NUMERIC_COLUMNS
x_raw = df[feature_cols].copy()


# Replace missing values with the median of each feature
# Median imputation prevents missing values from affecting PCA
imputer = SimpleImputer(strategy="median")

x_imputed = pd.DataFrame(
    imputer.fit_transform(x_raw),
    columns=feature_cols,
    index=x_raw.index
)


# Standardize features so all variables have comparable scale
# Each feature gets approximately mean = 0 and standard deviation = 1
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_imputed)

print(
    f"Loaded {len(df):,} students, "
    f"{len(feature_cols)} scaled numeric features"
)


# ---------- Apply PCA ----------

# Keep all principal components initially
# This allows us to study how much variance each component explains
pca_full = PCA(
    n_components=None,
    random_state=config.RANDOM_STATE
)

pca_full.fit(x_scaled)


# Percentage of total information explained by each component
explained_variance_ratio = pca_full.explained_variance_ratio_

# Running total of explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)


print(
    f"\nTotal components: "
    f"{len(explained_variance_ratio)}"
)


# Display variance explained by the first 10 components
print("\nVariance explained by first 10 components:")

for i in range(10):
    print(
        f"PC{i+1}: "
        f"{round(explained_variance_ratio[i] * 100, 2)}% "
        f"(Cumulative: "
        f"{round(cumulative_variance[i] * 100, 2)}%)"
    )


# Find the number of components needed for common variance levels
# Example: 90% means we want to retain 90% of the original information
for threshold in [0.80, 0.90, 0.95]:

    n_needed = (
        np.searchsorted(cumulative_variance, threshold) + 1
    )

    print(
        f"{int(threshold * 100)}% variance "
        f"requires {n_needed} components"
    )


# ---------- Create variance plots ----------

# Create two plots: individual and cumulative variance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Display at most the first 20 components
n_show = min(20, len(explained_variance_ratio))


# Scree plot shows variance explained by each component
axes[0].bar(
    range(1, n_show + 1),
    explained_variance_ratio[:n_show] * 100,
    color="#4C72B0"
)

axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_title("Scree Plot")


# Cumulative plot shows total variance retained
axes[1].plot(
    range(1, n_show + 1),
    cumulative_variance[:n_show] * 100,
    marker="o",
    color="#DD8452"
)


# Add reference lines for 80%, 90%, and 95% variance
for threshold in [80, 90, 95]:

    axes[1].axhline(
        y=threshold,
        color="gray",
        linestyle="--",
        linewidth=1
    )


axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].set_title("Cumulative Variance")


# Adjust spacing and save the figure
plt.tight_layout()

plt.savefig(
    f"{PCA_DIR}/scree_and_cumulative_variance.png"
)

plt.close()

print("Scree plot saved")


# ---------- Analyze PC1 and PC2 ----------

# PCA components contain weights called loadings
# Loadings show how strongly each original feature contributes
# to a principal component
loadings = pd.DataFrame(
    pca_full.components_[:2].T,
    columns=["PC1", "PC2"],
    index=feature_cols
)


# Find the 8 features with the largest contribution to PC1
top_pc1 = (
    loadings["PC1"]
    .abs()
    .sort_values(ascending=False)
    .head(8)
)


# Find the 8 features with the largest contribution to PC2
top_pc2 = (
    loadings["PC2"]
    .abs()
    .sort_values(ascending=False)
    .head(8)
)


print("\nTop features in PC1:")

for feature in top_pc1.index:

    print(
        feature,
        round(loadings.loc[feature, "PC1"], 3)
    )


print("\nTop features in PC2:")

for feature in top_pc2.index:

    print(
        feature,
        round(loadings.loc[feature, "PC2"], 3)
    )


# ---------- Create PCA loading biplot ----------

# The biplot shows how original features relate to PC1 and PC2
plt.figure(figsize=(8, 8))


for feature in feature_cols:

    # Draw an arrow representing the feature's loading
    plt.arrow(
        0,
        0,
        loadings.loc[feature, "PC1"],
        loadings.loc[feature, "PC2"],
        head_width=0.02,
        color="#6E2436",
        alpha=0.6
    )


    # Add the feature name near its arrow
    plt.text(
        loadings.loc[feature, "PC1"] * 1.1,
        loadings.loc[feature, "PC2"] * 1.1,
        feature,
        fontsize=7,
        ha="center"
    )


# Draw a unit circle to visualize loading strength
circle = plt.Circle(
    (0, 0),
    1,
    fill=False,
    linestyle="--",
    color="gray"
)

plt.gca().add_artist(circle)


plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.xlabel(
    f"PC1 ({round(explained_variance_ratio[0] * 100, 1)}%)"
)

plt.ylabel(
    f"PC2 ({round(explained_variance_ratio[1] * 100, 1)}%)"
)

plt.title("PCA Loadings Biplot")

# Keep the x and y axes at the same scale
plt.gca().set_aspect("equal")

plt.tight_layout()

plt.savefig(
    f"{PCA_DIR}/loadings_biplot.png"
)

plt.close()

print("Loadings biplot saved")


# ---------- Create 2D PCA projection ----------

# Calculate how much total variance is represented by PC1 and PC2
pc1_pc2_variance = cumulative_variance[1]

print(
    f"\nPC1 + PC2 capture "
    f"{round(pc1_pc2_variance * 100, 1)}% variance"
)


# Transform the original data into PCA space
# Keep only PC1 and PC2 for 2D visualization
x_2d = pca_full.transform(x_scaled)[:, :2]


# Create a scatter plot of students in PCA space
plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    x_2d[:, 0],
    x_2d[:, 1],

    # Color points using the actual placement outcome
    c=df[config.TARGET_COLUMN],

    cmap="coolwarm",
    s=5,
    alpha=0.4
)


plt.xlabel(
    f"PC1 ({round(explained_variance_ratio[0] * 100, 1)}%)"
)

plt.ylabel(
    f"PC2 ({round(explained_variance_ratio[1] * 100, 1)}%)"
)

plt.title(
    "PCA 2D Projection by Placement"
)


# Add a legend showing placement values
plt.colorbar(
    scatter,
    label="Placed (1) / Not Placed (0)"
)

plt.tight_layout()

plt.savefig(
    f"{PCA_DIR}/pca_2d_by_placement.png"
)

plt.close()

print("2D projection saved")


# ---------- Save PCA report ----------

# Store important PCA results in a text file
with open(
    f"{PCA_DIR}/pca_report.txt",
    "w"
) as f:

    f.write(
        f"PCA Report ({len(df)} students)\n\n"
    )


    # Save variance information
    f.write("Variance Explained:\n")

    for i in range(10):

        f.write(
            f"PC{i+1}: "
            f"{round(explained_variance_ratio[i] * 100, 2)}% "
            f"(Cumulative: "
            f"{round(cumulative_variance[i] * 100, 2)}%)\n"
        )


    # Save required component counts
    f.write("\nComponents Needed:\n")

    for threshold in [0.80, 0.90, 0.95]:

        n_needed = (
            np.searchsorted(
                cumulative_variance,
                threshold
            ) + 1
        )

        f.write(
            f"{int(threshold * 100)}% variance: "
            f"{n_needed} components\n"
        )


    # Save variance represented by the first two PCs
    f.write(
        f"\nPC1 + PC2 Variance: "
        f"{round(pc1_pc2_variance * 100, 1)}%\n"
    )


    # Save important PC1 features
    f.write("\nTop PC1 Features:\n")

    for feature in top_pc1.index:

        f.write(
            f"{feature}: "
            f"{round(loadings.loc[feature, 'PC1'], 3)}\n"
        )


    # Save important PC2 features
    f.write("\nTop PC2 Features:\n")

    for feature in top_pc2.index:

        f.write(
            f"{feature}: "
            f"{round(loadings.loc[feature, 'PC2'], 3)}\n"
        )


print(
    f"\nReport saved to "
    f"{PCA_DIR}/pca_report.txt"
)