from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from random_forest_regressor import RandomForestRegressor
from test_data import get_random_slice


def main():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    forest = RandomForestRegressor(n_estimators=10, max_depth=6)
    forest.fit(X, y)

    X_test, y_test = get_random_slice(X, y, size=10)

    y_pred = forest.predict(X_test)

    print("Predicted: ", y_pred)
    print("True: ", y_test)

    mse = mean_squared_error(y_test, y_pred)
    # Mean Squared Error (MSE): average of squared errors.  
    # Punishes large errors more strongly.  
    # Lower is better.  

    mae = mean_absolute_error(y_test, y_pred)
    # Mean Absolute Error (MAE): average of absolute errors.  
    # Represents the average deviation between predictions and true values.  
    # Easy to interpret in the original units (e.g. $). Lower is better.  

    r2 = r2_score(y_test, y_pred)
    # R-squared (R²): proportion of variance in the target explained by the model.  
    # 1.0 = perfect fit, 0 = no better than predicting the mean, <0 = worse than mean.  

    print("MSE :", mse)
    print("MAE :", mae)
    print("R²  :", r2)


if __name__ == "__main__":
    main()
