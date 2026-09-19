import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "cleaned_placement_data.csv")
PLOT_DIR = os.path.join(BASE_DIR, "output", "Plot")

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def run_kmeans(df):
    print("="*60)
    print("TOPIC 21: K-MEANS CLUSTERING")
    print("="*60)
    
    # Select numeric features for clustering
    features = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    df_cluster = df.dropna(subset=features).copy()
    X = df_cluster[features]
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal k using Elbow Method
    inertias = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k (Number of clusters)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.savefig(os.path.join(PLOT_DIR, "kmeans_elbow.png"))
    plt.close()
    
    # Choose k=3 based on assumed elbow
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_cluster['PCA1'] = X_pca[:, 0]
    df_cluster['PCA2'] = X_pca[:, 1]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=df_cluster, s=60)
    plt.title(f'K-Means Clustering (k={k}) Visualized with PCA')
    plt.savefig(os.path.join(PLOT_DIR, "kmeans_clusters.png"))
    plt.close()
    
    print(f"Performed K-Means clustering with k={k}.")
    print("Cluster sizes:")
    print(df_cluster['Cluster'].value_counts())
    
    # Cluster profiles
    print("\nCluster Profiles (Mean values):")
    profiles = df_cluster.groupby('Cluster')[features].mean()
    print(profiles)
    print("\nPlots saved: kmeans_elbow.png, kmeans_clusters.png")

def run_hierarchical(df):
    print("\n" + "="*60)
    print("TOPIC 24: HIERARCHICAL CLUSTERING")
    print("="*60)
    
    features = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    df_cluster = df.dropna(subset=features).copy()
    # Subsample for clear dendrogram
    df_sample = df_cluster.sample(n=50, random_state=42).copy()
    X = df_sample[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Linkage matrix
    Z = linkage(X_scaled, method='ward')
    
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=df_sample.index.astype(str).tolist(), leaf_rotation=90, leaf_font_size=8)
    plt.title('Hierarchical Clustering Dendrogram (Sample of 50 students)')
    plt.xlabel('Student Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "hierarchical_dendrogram.png"))
    plt.close()
    
    # Agglomerative Clustering
    agg_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
    df_sample['Agg_Cluster'] = agg_cluster.fit_predict(X_scaled)
    
    print("Performed Hierarchical (Agglomerative) Clustering.")
    print("Generated Dendrogram plot.")
    print("Plot saved: hierarchical_dendrogram.png")


if __name__ == "__main__":
    try:
        df = pd.read_csv(DATA_PATH)
        run_kmeans(df)
        run_hierarchical(df)
        print("\nUnsupervised experiments completed successfully!")
    except FileNotFoundError:
        print("Data file not found. Please run clean_data.py first.")
