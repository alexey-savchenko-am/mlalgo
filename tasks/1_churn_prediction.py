import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

#-------------------------------------------------------------------------------

"""
Problem: Predicting Customer Churn

You have a dataset with information about a telecom company's customers.
Features include:

-age, gender
-number of months with the service
-tariff plan
-average bill
-number of support requests
-debt status
-and a binary target: customer churn (1) or retention (0).

Required:
Build a baseline solution: load the data, split the train/test, train a simple model (e.g., logistic regression).
Choose quality metrics: accuracy isn't always appropriate here. 
You need to justify whether ROC-AUC, F1-score, or precision/recall is better (depending on the business context).
Improve the model: try boosting, hyperparameter selection, and feature engineering 
(e.g., aggregate expenses for recent months, extract categorical features).
Show interpretation: which features most influence the likelihood of churn.
Describe how to deploy the model: for example, package it into a FastAPI service and provide it to the business as an API.

"""
data = pd.DataFrame({
    "age": [25, 45, 33, 52, 40, 29, 60, 31, 50, 42,
            36, 27, 48, 55, 39, 30, 62, 41, 28, 49],
    "months_on_service": [3, 24, 12, 36, 18, 6, 48, 8, 30, 20,
                          15, 4, 27, 40, 14, 5, 50, 19, 7, 32],
    "avg_bill": [20, 70, 50, 90, 60, 25, 100, 35, 80, 65,
                 55, 22, 75, 95, 58, 28, 110, 62, 26, 85],
    "support_calls": [1, 5, 2, 6, 3, 1, 7, 2, 4, 3,
                      2, 1, 5, 6, 3, 2, 8, 3, 1, 6],
    "has_debt": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    "churn": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
              0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
})

#-------------------------------- 1/DATA SET ANALYSIS-----------------------------------
"""
Question: Tell me what you'll do first with this dataset and why.

"First, I'll look at the basic information about the dataset (.info(), .describe()),
check the class balance in the target variable churn, and look at the feature distribution.
This will help me understand whether there are imbalances, gaps, or anomalies, and whether they need to be addressed."
"""
data.info()
data.describe()
print(f"Distribution of features: {data['churn'].value_counts(normalize=True)}")

print("CORRELATIONS")
print(data.corr(numeric_only=True))
"""
  The value of the has_debt column is highly correlated with churn (1.0), so it will have to be removed.
"""

#----------------------------------- 2/DATA SPLIT -----------------------------------------------
"""
Question: How would you split the data into train and test (or validation)?

"If the data is static, I'll split it randomly, taking into account target stratification,
to maintain class balance. If the data is time-based, I'll do a time-based split:
train = past periods, test = future, to avoid information leakage."
"""

X: DataFrame = data.drop(columns=["has_debt", "churn"])
y: Series = data['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, # test sample 20%
    stratify=y, # equal proportion
    random_state=42)

"""
"I use train_test_split with test_size=0.2 and stratify=y to maintain class balance, 
and I fix random_state for reproducibility."
"""

#-------------------------------------3/Baseline Logistic Regression---------------------------

# fitting
model = LogisticRegression()
model.fit(X_train, y_train)

"""
"L1 regularization reset the coefficients of two features to zero, effectively eliminating them.
This confirms that the features in the dataset are highly correlated with each other, and the model retained a minimal set.
This can be used to combat multicollinearity and perform feature selection."
"""
model_l1 = LogisticRegression(penalty="l1", C=0.1, solver="liblinear")
model_l1.fit(X_train, y_train)

coef_table = pd.DataFrame({
    "feature": X.columns,
    "coef": model_l1.coef_[0]
})

print(coef_table)

#prediction

"""
"I use both predict and predict_proba. 
The former is needed for metrics like accuracy or confusion matrix. 
The latter is for probabilities, to build ROC-AUC and manage the classification threshold. 
In a real-world problem, a business might say, 
'We need fewer false negatives,' and then I use predict_proba to shift the threshold."
"""
y_pred = model_l1.predict(X_test)

# first column [:, 0] → probability that the object belongs to class 0 (e.g. "the client will stay").
# second column [:, 1] → probability that the object belongs to class 1 (for example, “the client will leave”).
y_proba = model_l1.predict_proba(X_test)[:, 1]

print("REAL DATA VS PREDICTED")
for real, pred in zip(y_test.values, y_pred):
    print(f"real: {real} : predicted {pred}")

#---------------------------------------METRICKS------------------------------------------------------

"""
"After training, I calculate accuracy, F1, and ROC-AUC. Accuracy measures overall quality, 
F1 takes into account the error balance, and ROC-AUC is useful for probability analysis and threshold selection. 
I also look at the confusion matrix to understand which features the model is most likely to confuse."
"""

#Accuracy — the proportion of correct predictions
print("Accuracy: ", accuracy_score(y_test, y_pred))

# F1 — precision и recall ballance
print("F1-score:", f1_score(y_test, y_pred))

# ROC-AUC — quality of ranking by probability
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# Confusion Matrix
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))