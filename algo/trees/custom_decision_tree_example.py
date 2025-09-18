from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

mse = mean_squared_error(y_true, y_pred)
# Mean Squared Error (MSE): average of squared errors.  
# Punishes large errors more strongly.  
# Lower is better.  

mae = mean_absolute_error(y_true, y_pred)
# Mean Absolute Error (MAE): average of absolute errors.  
# Represents the average deviation between predictions and true values.  
# Easy to interpret in the original units (e.g. $). Lower is better.  

r2 = r2_score(y_true, y_pred)
# R-squared (R²): proportion of variance in the target explained by the model.  
# 1.0 = perfect fit, 0 = no better than predicting the mean, <0 = worse than mean.  

print("MSE :", mse)
print("MAE :", mae)
print("R²  :", r2)