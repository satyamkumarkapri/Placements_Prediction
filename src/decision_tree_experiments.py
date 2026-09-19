import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "cleaned_placement_data.csv")
PLOT_DIR = os.path.join(BASE_DIR, "output", "Plot")
REPORT_DIR = os.path.join(BASE_DIR, "output", "Report")

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

def calculate_entropy(y):
    p = np.bincount(y) / len(y)
    return -np.sum([p_i * np.log2(p_i) for p_i in p if p_i > 0])

def calculate_gini(y):
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p**2)

def main():
    print("==========================================================")
    print("   DECISION TREES & PRUNING EXPERIMENTS (TOPICS 13-15)    ")
    print("==========================================================\n")

    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Error: Dataset not found. Please ensure data is available.")
        return

    features = [
        "SGPA_Sem8", "MockInterviewScore", "Projects", 
        "SoftSkillsRating", "Certifications", "CGPA", 
        "AptitudeTestScore", "Internships"
    ]
    df = df.dropna(subset=features + ["PlacementStatus"])
    X = df[features]
    y = df["PlacementStatus"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    metrics = {}

    print("\n" + "="*50)
    print("13. BASIC DECISION TREE")
    print("="*50)
    
    # 1. Unconstrained Tree
    dt_full = DecisionTreeClassifier(random_state=42)
    dt_full.fit(X_train, y_train)
    metrics["unconstrained_train_acc"] = float(dt_full.score(X_train, y_train))
    metrics["unconstrained_val_acc"] = float(dt_full.score(X_test, y_test))
    metrics["unconstrained_depth"] = int(dt_full.get_depth())
    
    print(f"Unconstrained Tree - Train Accuracy: {metrics['unconstrained_train_acc']:.4f} (Likely Overfitting)")
    print(f"Unconstrained Tree - Test Accuracy:  {metrics['unconstrained_val_acc']:.4f}")
    print(f"Tree Depth: {metrics['unconstrained_depth']}")

    # 2. Shallow Tree
    dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_shallow.fit(X_train, y_train)
    metrics["shallow_train_acc"] = float(dt_shallow.score(X_train, y_train))
    metrics["shallow_val_acc"] = float(dt_shallow.score(X_test, y_test))
    
    print(f"\nShallow Tree (max_depth=3) - Train Accuracy: {metrics['shallow_train_acc']:.4f}")
    print(f"Shallow Tree (max_depth=3) - Test Accuracy:  {metrics['shallow_val_acc']:.4f}")

    # Feature Importances
    importances = dt_shallow.feature_importances_
    top_indices = np.argsort(importances)[::-1][:5]
    top_features = [{"name": features[i], "value": float(importances[i])} for i in top_indices if importances[i] > 0]
    metrics["top_features"] = top_features

    # Plot tree with high DPI and big figure size
    plt.figure(figsize=(24, 14), dpi=300)
    plot_tree(dt_shallow, feature_names=features, class_names=["Not Placed", "Placed"], filled=True, rounded=True, proportion=False, fontsize=10)
    plt.title("Decision Tree (max_depth=3)", fontsize=20)
    plt.savefig(os.path.join(PLOT_DIR, "dt_tree_plot.png"), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6), dpi=150)
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], align='center', color='#3b82f6')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Feature Importance (shallow tree)')
    plt.savefig(os.path.join(PLOT_DIR, "dt_feature_importance.png"), bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("14. PRUNING AND SPLITTING CRITERIA")
    print("="*50)
    
    # Session 14 Metrics
    metrics["root_entropy"] = float(calculate_entropy(y_train))
    metrics["root_gini"] = float(calculate_gini(y_train))
    
    dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=42)
    dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)
    dt_gini.fit(X_train, y_train)
    dt_entropy.fit(X_train, y_train)
    
    metrics["gini_acc"] = float(dt_gini.score(X_test, y_test))
    metrics["entropy_acc"] = float(dt_entropy.score(X_test, y_test))
    
    print("--- Splitting Criteria: Gini vs Entropy ---")
    print(f"Gini Criterion Test Accuracy:    {metrics['gini_acc']:.4f}")
    print(f"Entropy Criterion Test Accuracy: {metrics['entropy_acc']:.4f}")

    # Pruning
    print("\n--- Cost Complexity Pruning (Post-Pruning) ---")
    path = dt_full.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    
    # Subsample alphas for speed but ensure we find a good one
    if len(ccp_alphas) > 50:
        alphas_to_test = ccp_alphas[::max(1, len(ccp_alphas)//50)]
    else:
        alphas_to_test = ccp_alphas
        
    train_scores = []
    test_scores = []
    for alpha in alphas_to_test:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        clf.fit(X_train, y_train)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))

    best_idx = np.argmax(test_scores)
    best_alpha = alphas_to_test[best_idx]
    best_pruned_acc = test_scores[best_idx]
    
    metrics["best_ccp_alpha"] = float(best_alpha)
    metrics["best_pruned_acc"] = float(best_pruned_acc)
    
    print(f"Best CCP Alpha: {best_alpha:.5f}")
    print(f"Pruned Tree Test Accuracy: {best_pruned_acc:.4f}")

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(alphas_to_test, train_scores, marker='o', label='Train accuracy', drawstyle="steps-post", color='#3b82f6')
    plt.plot(alphas_to_test, test_scores, marker='o', label='Validation accuracy', drawstyle="steps-post", color='#f59e0b')
    plt.axvline(x=best_alpha, color='gray', linestyle='--', label=f'Best alpha ({best_alpha:.5f})')
    plt.xlabel("ccp_alpha")
    plt.ylabel("Accuracy")
    plt.title("Pruning: Accuracy vs ccp_alpha")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "dt_pruning.png"), bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("15. BIAS-VARIANCE TRADEOFF VIA TREE DEPTH")
    print("="*50)
    
    # Session 15 Metrics
    depths = list(range(1, 26))
    depth_stats = []
    train_scores_depth = []
    test_scores_depth = []
    
    print(f"{'Max Depth':<12} | {'Train Acc':<12} | {'Test Acc':<12} | {'Gap'}")
    print("-" * 60)
    
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        tr = clf.score(X_train, y_train)
        te = clf.score(X_test, y_test)
        gap = tr - te
        
        train_scores_depth.append(tr)
        test_scores_depth.append(te)
        depth_stats.append({
            "depth": d,
            "train_acc": float(tr),
            "val_acc": float(te),
            "gap": float(gap)
        })
        
        if d in [1, 2, 4, 6, 8, 10, 15, 20, 25]:
            print(f"{d:<12} | {tr:<12.4f} | {te:<12.4f} | {gap:+.4f}")
        
    best_depth_idx = np.argmax(test_scores_depth)
    best_depth = depths[best_depth_idx]
    best_depth_acc = test_scores_depth[best_depth_idx]
    
    metrics["depth_stats"] = depth_stats
    metrics["best_depth"] = int(best_depth)
    metrics["best_depth_acc"] = float(best_depth_acc)

    plt.figure(figsize=(12, 6), dpi=150)
    plt.plot(depths, train_scores_depth, marker='o', label='Train accuracy', color='#3b82f6')
    plt.plot(depths, test_scores_depth, marker='o', label='Validation accuracy', color='#10b981')
    plt.axvline(x=best_depth, color='gray', linestyle='--', label=f'Best depth = {best_depth}')
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Bias-Variance Tradeoff: Accuracy vs Tree Depth")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "dt_bias_variance.png"), bbox_inches='tight')
    plt.close()

    # Ensemble Methods (Topics 16, 17, 18, 20)
    run_ensemble_methods(X_train, X_test, y_train, y_test, metrics)

    # Save metrics to JSON
    with open(os.path.join(REPORT_DIR, "dt_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nGenerated all plots and metrics successfully to output/Plot and output/Report!")
    print("Experiment Complete. Review the UI to see practical implementations.")

def run_ensemble_methods(X_train, X_test, y_train, y_test, metrics):
    print("\n" + "="*50)
    print("16. RANDOM FORESTS AND BAGGING")
    print("="*50)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_train_acc = rf.score(X_train, y_train)
    rf_test_acc = rf.score(X_test, y_test)
    
    metrics["rf_train_acc"] = float(rf_train_acc)
    metrics["rf_test_acc"] = float(rf_test_acc)
    
    print(f"Random Forest Train Accuracy: {rf_train_acc:.4f}")
    print(f"Random Forest Test Accuracy:  {rf_test_acc:.4f}")

    print("\n" + "="*50)
    print("17 & 18. BOOSTING BASICS & MODERN BOOSTED TREES (XGBOOST)")
    print("="*50)
    
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_train_acc = xgb.score(X_train, y_train)
    xgb_test_acc = xgb.score(X_test, y_test)

    metrics["xgb_train_acc"] = float(xgb_train_acc)
    metrics["xgb_test_acc"] = float(xgb_test_acc)
    
    print(f"XGBoost Train Accuracy: {xgb_train_acc:.4f}")
    print(f"XGBoost Test Accuracy:  {xgb_test_acc:.4f}")
    
    print("\nNote: Topic 19 (Feature Importance and SHAP) is demonstrated in src/train_model.py")
    
    print("\n" + "="*50)
    print("20. COMPARING MODEL TRACKS")
    print("="*50)
    
    print(f"{'Model':<15} | {'Test Accuracy'}")
    print("-" * 35)
    print(f"{'Decision Tree':<15} | {metrics.get('best_pruned_acc', metrics.get('shallow_val_acc', 0)):.4f}")
    print(f"{'Random Forest':<15} | {rf_test_acc:.4f}")
    print(f"{'XGBoost':<15} | {xgb_test_acc:.4f}")

if __name__ == "__main__":
    main()
