from typing import Optional, Tuple
import numpy as np
import pickle

Array = np.ndarray

class DecisionTreeRegressor:

    class Node:
        def __init__(
            self,
            feature: Optional[int] = None,
            threshold: Optional[float] = None,
            left: Optional["DecisionTreeRegressor.Node"] = None,
            right: Optional["DecisionTreeRegressor.Node"] = None,
            value: Optional[float] = None
        ) -> None:
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
    
    def __init__(self, max_depth: int = 2) -> None:
        self.max_depth = max_depth
        self.root: Optional[DecisionTreeRegressor.Node] = None


    # --------------------------------- public methods ---------------------------------
    def fit(self, X: Array, y: Array) -> None:
        """Build a decision tree regressor from training data."""
        self.root = self._build_tree(X, y)

    def predict(self, X: Array) -> Array:
        """Predict target values for given samples X."""
        if self.root is None:
            raise ValueError("Tree is not fitted yet.")
        return np.array([self._predict_one(self.root, x) for x in X])
    
    def print_tree(self) -> None:
        """Print the whole tree structure from the root."""
        if self.root is None:
            print("Tree is empty")
            return

        def _print(node: DecisionTreeRegressor.Node, depth: int = 0) -> None:
            """Recursive helper for printing a subtree."""
            indent = "  " * depth
            if node.value is not None:
                print(f"{indent}Leaf: value={node.value:.2f}")
            else:
                print(f"{indent}Feature {node.feature} <= {node.threshold:.2f}?")
                _print(node.left, depth + 1)   # type: ignore
                _print(node.right, depth + 1)  # type: ignore

        _print(self.root, 0)

    def save(self, path: str, use_joblib: bool = True) -> None:
        """
        Save the trained model to disk.
        """
        if use_joblib:
            try:
                import joblib
                joblib.dump(self, path)
                return
            except Exception:
                #fallback to pickle if joblib isn't available of failed
                pass
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "DecisionTreeRegressor":
        """
        Load a saved model from disk. Returns the DecisionTreeRegressor instance.
        Accepts files written by joblib or pickle.
        """
        try:
            import joblib
            obj = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                obj = pickle.load(f)

        # optional: check that loaded object is of expected type
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__} instance (got {type(obj)})")
        return obj

    # --------------------------------- private methods ---------------------------------
    def _mse(self, y: Array) -> float:
        return float(np.var(y)) * len(y)
    
    def _split_dataset(
            self, X: Array, y: Array, feature_index: int, threshold: float
    ) -> Tuple[Array, Array, Array, Array]:
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]
    
    def _best_split(self, X: Array, y: Array) -> Tuple[Optional[int], Optional[float]]:
        best_feature, best_threshold = None, None
        best_score = float("inf")
        n_samples, n_features = X.shape

        for feature in range(n_features):
            values = np.unique(X[:, feature])
            thresholds = (values[:-1] + values[1:]) / 2
            for t in thresholds:
                X_left, y_left, X_right, y_right = self._split_dataset(X, y, feature, t)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                score = self._mse(y_left) + self._mse(y_right)
                if score < best_score:
                    best_feature, best_threshold, best_score = feature, t, score
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: Array, y: Array, depth: int = 0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return self.Node(value=float(np.mean(y)))
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return self.Node(value=float(np.mean(y)))
        
        X_left, y_left, X_right, y_right = self._split_dataset(X, y, feature, threshold)
        left_node = self._build_tree(X_left, y_left, depth+1)
        right_node = self._build_tree(X_right, y_right, depth+1)

        return self.Node(feature=feature, threshold=threshold, left=left_node, right=right_node)
    
    def _predict_one(self, node: Node, x: Array) -> float:
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x) 



    
            