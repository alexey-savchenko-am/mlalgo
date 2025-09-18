from sklearn.datasets import fetch_california_housing
from decision_tree_regressor import DecisionTreeRegressor
import numpy as np
from typing import Dict, List

housing = fetch_california_housing()

X = housing.data
y = housing.target
features = housing.feature_names

#tree = DecisionTreeRegressor(max_depth=6)
#tree.fit(X, y)
#tree.save("tree_model.joblib") 

tree = DecisionTreeRegressor.load("tree_model.joblib")

tree.print_tree()

rng = np.random.default_rng(42)
idx = rng.choice(len(X), size=5, replace=False)

X_test = X[idx]
y_true = y[idx]

y_pred = tree.predict(X_test)

print("Predicted:", y_pred)
print("True:", y_true)