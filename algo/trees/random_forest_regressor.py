from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from typing import List
from decision_tree_regressor import DecisionTreeRegressor

Array = np.ndarray

class RandomForestRegressor:
    def __init__(self, n_estimators: int = 10, max_depth: int = 5, bootstrap: bool = True, n_jobs: int = -1) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs  # -1 = use all cores
        self.trees: List[DecisionTreeRegressor] = []
    
    def fit(self, X: Array, y: Array) -> None:
        """Builds random forrest concurrently"""
        seeds = np.random.SeedSequence().spawn(self.n_estimators)
        with ProcessPoolExecutor(max_workers=None if self.n_jobs == -1 else self.n_jobs) as executor:
            futures = [
                executor.submit(self._fit_single_tree, X, y, int(seed.generate_state(1)[0])) for seed in seeds
            ]

            for future in as_completed(futures):
                self.trees.append(future.result())
    
    def predict(self, X: Array) -> Array:
        """Average prediction of all trees"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return predictions.mean(axis=0)
        
    
    def _fit_single_tree(self, X: Array, y: Array, seed: int) -> DecisionTreeRegressor:
        """Function for fitting a single tree"""
        rng = np.random.default_rng(seed)
        n_samples = X.shape[0]

        if self.bootstrap:
            indices = rng.integers(0, n_samples, n_samples)
            X_samples, y_samples = X[indices], y[indices]
        else:
            X_samples, y_samples = X, y

        tree = DecisionTreeRegressor(max_depth=self.max_depth)
        tree.fit(X_samples, y_samples)
        return tree