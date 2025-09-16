
from typing import Optional, Tuple, List
import numpy as np

Array = np.ndarray

class Node:
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        value: Optional[float] = None
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def mse(y: Array) -> float:
    """Sum of squared errors (proxy for split quality)"""
    return float(np.var(y) * len(y))

def split_dataset(
        X: Array, y: Array, feature_index: int, threshold: float
) -> Tuple[Array, Array, Array, Array]:
    """ Splits dataset by feature and treshold """
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def best_split(X: Array, y: Array) -> Tuple[Optional[int], Optional[float]]:
    """"""
    best_feature: Optional[int] = None
    best_threshold: Optional[float] = None
    best_score: float = float("inf")
    n_samples, n_features = X.shape

    for feature in range(n_features):
        """ list of possible tresholds by the feature"""
        values = np.unique(X[:, feature])

        tresholds = (values[:-1] + values[1:]) / 2

        for t in tresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature, t)
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            score = mse(y_left) + mse(y_right)

            if score < best_score:
                best_feature, best_threshold, best_score = feature, t, score

    return best_feature, best_threshold

def build_tree(
        X: Array, y: Array, depth: int = 0, max_depth: int = 2
) -> Node:
     if depth >= max_depth or len(np.unique(y)) == 1:
         return Node(value=float(np.mean(y)))
     
     feature, threshold = best_split(X, y)
     if feature is None:
         return Node(value=float(np.mean(y)))
     
     X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)

     left_node = build_tree(X_left, y_left, depth + 1, max_depth)
     right_node = build_tree(X_right, y_right, depth + 1, max_depth)

     return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

def predict_one(node: Node, x: Array) -> float:
    if node.value is not None:
        return node.value
    if node.feature is None or node.threshold is None:
        raise ValueError("Некорректный узел дерева")
    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)  # type: ignore
    else:
        return predict_one(node.right, x)  # type: ignore


def predict(tree: Node, X: Array) -> Array:
    return np.array([predict_one(tree, x) for x in X])

def print_tree(node: Node, depth: int = 0):

    indent = "  " * depth

    if node.value is not None:
        print(f"{indent}Leaf: value={node.value:.2f}")
    else:
        print(f"{indent}Feature {node.feature} <= {node.threshold:.2f}?")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)

X = np.array([
    [40, 2000],
    [50, 1995],
    [60, 2010],
    [70, 2005],
    [80, 2015],
    [90, 2020],
])

y = np.array([100, 120, 150, 160, 200, 220])

root = build_tree(X, y)
print_tree(root, 0)
