import sys, os

# Add project root folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib

# Use non-GUI backend for saving plots
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src import config

# Create DBSCAN output folder if it doesn't exist
os.makedirs(config.DBSCAN_DIR, exist_ok=True)

# Number of students used for DBSCAN
SUBSAMPLE_SIZE = 5000

# ---------- Load, impute, scale ----------

# Load cleaned dataset
df_full = pd.read_csv(config.CLEAN_DATA_PATH)

# Get numeric feature columns
feature_cols = config.NUMERIC_COLUMNS

# Select only numeric features
x_raw_full = df_full[feature_cols].copy()

# Fill missing values using median
imputer = SimpleImputer(strategy="median")
x_imputed_full = pd.DataFrame(
    imputer.fit_transform(x_raw_full),
    columns=feature_cols,
    index=x_raw_full.index
)

# Standardize features to mean=0 and std=1
scaler = StandardScaler()
x_scaled_full = scaler.fit_transform(x_imputed_full)

# Create random generator
rng = np.random.RandomState(config.RANDOM_STATE)

# Randomly select 5000 students
sample_idx = rng.choice(
    len(x_scaled_full),
    size=SUBSAMPLE_SIZE,
    replace=False
)

# Use only selected students for DBSCAN
x_scaled = x_scaled_full[sample_idx]

# Keep corresponding student records
df = df_full.iloc[sample_idx].reset_index(drop=True)

print(
    f"Loaded {len(df_full):,} students, "
    f"subsampled {SUBSAMPLE_SIZE:,} for DBSCAN"
)


# Check whether anomaly labels exist
if 'IsAnomaly' in df.columns:

    print(
        f"IsAnomaly flag present: "
        f"{df['IsAnomaly'].sum():,} anomalous students"
    )

    # Anomaly label is NOT used for DBSCAN training
    print(
        "(IsAnomaly was not given to DBSCAN; "
        "it is only used for comparison)"
    )

else:
    print("No IsAnomaly flag found.")


# ---------- Choosing eps ----------

# Set min_samples as 2 × number of features
MIN_SAMPLES = 2 * len(feature_cols)

# Find nearest neighbors
neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES)

# Train nearest-neighbor model
neighbors.fit(x_scaled)

# Calculate distances to nearest neighbors
distances, _ = neighbors.kneighbors(x_scaled)

# Sort distance to kth nearest neighbor
k_distances = np.sort(distances[:, -1])


# Create k-distance plot
plt.figure(figsize=(9, 5))
plt.plot(k_distances)

plt.xlabel("Points sorted by k-distance")
plt.ylabel(f"Distance to {MIN_SAMPLES}-th nearest neighbor")
plt.title("K-Distance Plot for Choosing eps")

plt.grid(alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig(f"{config.DBSCAN_DIR}/k_distance_plot.png")
plt.close()

print(f"\nK-distance plot saved")


# ---------- Find eps automatically ----------

# Number of points
n_points = len(k_distances)

# Normalize x-axis
x_norm = np.linspace(0, 1, n_points)

# Normalize distances
y_norm = (
    (k_distances - k_distances.min())
    / (k_distances.max() - k_distances.min())
)

# First point of the curve
x1, y1 = x_norm[0], y_norm[0]

# Last point of the curve
x2, y2 = x_norm[-1], y_norm[-1]

# Calculate distance from each point to straight line
numerator = np.abs(
    (y2 - y1) * x_norm
    - (x2 - x1) * y_norm
    + x2 * y1
    - y2 * x1
)

denominator = np.sqrt(
    (y2 - y1) ** 2
    + (x2 - x1) ** 2
)

# Perpendicular distance from line
perpendicular_distance = numerator / denominator

# Find point with maximum distance = elbow
elbow_idx = int(np.argmax(perpendicular_distance))

# Use elbow distance as eps
suggested_eps = round(
    float(k_distances[elbow_idx]), 2
)

print(f"Suggested eps: {suggested_eps}")


# ---------- Run DBSCAN ----------

# Use automatically selected eps
EPS = suggested_eps

# Create DBSCAN model
dbscan = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES,
    algorithm="ball_tree",
    n_jobs=1
)

# Perform clustering
cluster_labels = dbscan.fit_predict(x_scaled)

# Count clusters (-1 represents noise)
n_clusters = len(set(cluster_labels)) - (
    1 if -1 in cluster_labels else 0
)

# Count noise points
n_noise = int((cluster_labels == -1).sum())

# Count core points
n_core = len(dbscan.core_sample_indices_)

print(f"\nDBSCAN (eps={EPS}, min_samples={MIN_SAMPLES}):")
print(f"Clusters found: {n_clusters}")
print(f"Core points: {n_core:,}")
print(f"Noise points: {n_noise:,}")


# Show size of each cluster
cluster_sizes = (
    pd.Series(cluster_labels)
    .value_counts()
    .sort_index()
)

print("\nCluster sizes (-1 = noise):")
print(cluster_sizes.to_string())


# ---------- Compare DBSCAN noise with anomaly flag ----------

# Initialize counters
both_flagged = 0
noise_that_is_anomaly = 0
anomaly_that_is_noise = 0

# Identify DBSCAN noise points
is_noise = cluster_labels == -1

if 'IsAnomaly' in df.columns:

    # Convert anomaly column to Boolean
    anomaly_flag = df["IsAnomaly"].values.astype(bool)

    # Points detected as both noise and anomaly
    both_flagged = int(
        (is_noise & anomaly_flag).sum()
    )

    # Percentage of noise points that are anomalies
    noise_that_is_anomaly = round(
        both_flagged / max(n_noise, 1) * 100,
        1
    )

    # Percentage of anomalies detected as noise
    anomaly_that_is_noise = round(
        both_flagged / max(anomaly_flag.sum(), 1) * 100,
        1
    )

    print("\nOverlap between DBSCAN noise and anomalies:")
    print(f"Both noise and anomalous: {both_flagged:,}")
    print(
        f"Noise points that are anomalies: "
        f"{noise_that_is_anomaly}%"
    )
    print(
        f"Anomalies detected as noise: "
        f"{anomaly_that_is_noise}%"
    )


# ---------- Test different eps values ----------

# Try smaller, normal and larger eps
eps_values_to_try = [
    round(EPS * 0.5, 2),
    EPS,
    round(EPS * 2, 2)
]

sensitivity_results = []

for eps_val in eps_values_to_try:

    # Create DBSCAN with current eps
    db = DBSCAN(
        eps=eps_val,
        min_samples=MIN_SAMPLES,
        algorithm="ball_tree",
        n_jobs=1
    )

    # Perform clustering
    labels = db.fit_predict(x_scaled)

    # Count clusters
    n_clust = len(set(labels)) - (
        1 if -1 in labels else 0
    )

    # Count noise
    n_ns = int((labels == -1).sum())

    # Store results
    sensitivity_results.append({
        "eps": eps_val,
        "n_clusters": n_clust,
        "n_noise": n_ns
    })

    print(
        f"eps={eps_val}: "
        f"{n_clust} clusters, "
        f"{n_ns:,} noise points"
    )


# ---------- Visualize using PCA ----------

# PCA is used only to visualize high-dimensional data in 2D
pca = PCA(
    n_components=2,
    random_state=config.RANDOM_STATE
)

# Convert data into 2 principal components
x_2d = pca.fit_transform(x_scaled)

plt.figure(figsize=(9, 7))

# Identify noise points
noise_mask = cluster_labels == -1

# Plot clustered points
if (~noise_mask).any():

    plt.scatter(
        x_2d[~noise_mask, 0],
        x_2d[~noise_mask, 1],
        c=cluster_labels[~noise_mask],
        cmap="tab10",
        s=6,
        alpha=0.5,
        label="Clustered"
    )

# Plot noise points
if noise_mask.any():

    plt.scatter(
        x_2d[noise_mask, 0],
        x_2d[noise_mask, 1],
        c="black",
        marker="x",
        s=15,
        alpha=0.6,
        label=f"Noise ({n_noise:,} points)"
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("DBSCAN Clusters and Noise")

plt.legend()
plt.tight_layout()

# Save visualization
plt.savefig(
    f"{config.DBSCAN_DIR}/dbscan_clusters_pca_2d.png"
)

plt.close()

print("\n2D PCA visualization saved")


# ---------- Placement rate ----------

# Compare placement rate of clusters vs noise
if config.TARGET_COLUMN in df.columns:

    # Placement rate among clustered students
    placement_clustered = (
        df.loc[
            ~noise_mask,
            config.TARGET_COLUMN
        ].mean()
        if (~noise_mask).any()
        else float("nan")
    )

    # Placement rate among noise students
    placement_noise = (
        df.loc[
            noise_mask,
            config.TARGET_COLUMN
        ].mean()
        if noise_mask.any()
        else float("nan")
    )

    print(
        f"\nPlacement rate, clustered points: "
        f"{round(placement_clustered * 100, 1)}%"
    )

    print(
        f"Placement rate, noise points: "
        f"{round(placement_noise * 100, 1)}%"
    )


# ---------- Save report ----------

# Create and save DBSCAN report
with open(
    f"{config.DBSCAN_DIR}/dbscan_report.txt",
    "w"
) as f:

    # Write basic information
    f.write(
        f"DBSCAN on {len(df):,} students\n"
    )

    f.write(
        f"{len(feature_cols)} scaled numeric features\n\n"
    )

    # Save DBSCAN parameters
    f.write(
        f"min_samples = {MIN_SAMPLES}\n"
    )

    f.write(
        f"eps = {EPS}\n\n"
    )

    # Save clustering results
    f.write(
        f"Clusters found: {n_clusters}\n"
    )

    f.write(
        f"Core points: {n_core:,}\n"
    )

    f.write(
        f"Noise points: {n_noise:,}\n\n"
    )

    # Save cluster sizes
    f.write("Cluster sizes (-1 = noise):\n")
    f.write(cluster_sizes.to_string())

    # Save anomaly comparison
    if 'IsAnomaly' in df.columns:

        f.write(
            "\n\nOverlap with IsAnomaly flag:\n"
        )

        f.write(
            f"Both noise and anomalous: "
            f"{both_flagged:,}\n"
        )

        f.write(
            f"Noise that is anomalous: "
            f"{noise_that_is_anomaly}%\n"
        )

        f.write(
            f"Anomalies detected as noise: "
            f"{anomaly_that_is_noise}%\n"
        )

    # Save eps sensitivity results
    f.write("\neps sensitivity:\n")

    for r in sensitivity_results:

        f.write(
            f"eps={r['eps']}: "
            f"{r['n_clusters']} clusters, "
            f"{r['n_noise']:,} noise points\n"
        )


print(
    f"\nReport saved to "
    f"{config.DBSCAN_DIR}/dbscan_report.txt"
)