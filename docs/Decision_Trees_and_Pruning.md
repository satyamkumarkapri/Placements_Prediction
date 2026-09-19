# Decision Trees and Pruning

This document covers all the essential topics related to Decision Trees, Pruning and Splitting Criteria, and the Bias-Variance Tradeoff via Tree Depth, integrating concepts from your presentation into this machine learning project.

## 13. Decision Trees
A Decision Tree is a supervised machine learning algorithm that can be used for both classification and regression problems. It operates by splitting the dataset into smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with **decision nodes** and **leaf nodes**.
- **Root Node:** Represents the entire population or sample and this further gets divided into two or more homogeneous sets.
- **Decision Node:** A sub-node that splits into further sub-nodes.
- **Leaf/Terminal Node:** Nodes that do not split.

In our project (`src/train_model.py`), we use the ensemble version of decision trees: the `RandomForestClassifier` and `RandomForestRegressor`.

## 14. Pruning and Splitting Criteria
### Splitting Criteria
The algorithm needs a way to evaluate which feature and threshold provides the best split at each node.
1. **Gini Impurity (`criterion='gini'`):** A measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. Lower Gini indicates a purer node.
2. **Entropy (`criterion='entropy'`):** A measure of the randomness in the information being processed. The split that provides the highest **Information Gain** (reduction in entropy) is chosen.

*Both generally produce similar trees, but Gini is slightly faster to compute since it doesn't involve logarithmic functions.*

### Pruning
Decision Trees are prone to overfitting. If left unconstrained, a tree will continue to split until every leaf is pure, creating a complex tree that perfectly memorizes the training data but fails to generalize to new data.
- **Pre-Pruning (Early Stopping):** Stopping the tree's growth before it perfectly classifies the training set (e.g., setting `max_depth`, `min_samples_split`, or `min_samples_leaf`).
- **Post-Pruning (Cost Complexity Pruning):** Growing the tree fully and then removing branches that do not provide significant predictive power. Controlled by the parameter `ccp_alpha`. A higher `ccp_alpha` increases the penalty for a larger tree, leading to more pruning.

## 15. Bias-Variance Tradeoff via Tree Depth
The depth of a decision tree heavily influences the **Bias-Variance Tradeoff**:
- **Underfitting (High Bias, Low Variance):** If the tree is too shallow (e.g., `max_depth=2`), it lacks the complexity to capture the underlying patterns in the data. The model is too simple.
- **Overfitting (Low Bias, High Variance):** If the tree is too deep (e.g., `max_depth=None`), it captures the noise in the training data perfectly. The training accuracy will approach 100%, but the test accuracy will suffer.
- **The Sweet Spot:** The goal is to find the optimal tree depth where the model captures the true relationships without memorizing the noise, leading to the highest generalization accuracy on the test set.

## Trying the Experiments
You can run the provided script `src/decision_tree_experiments.py` to see these concepts in action on our placement dataset. The script will demonstrate:
1. Training an unconstrained basic tree (which will overfit).
2. Comparing Gini and Entropy splitting criteria.
3. Applying Cost Complexity Pruning to find the optimal tree.
4. Experimenting with various tree depths to clearly illustrate the Bias-Variance tradeoff.
