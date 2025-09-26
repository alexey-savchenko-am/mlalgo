import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#-------------------------------------------------------------------------------
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
    Вопрос: Скажи, что ты сделаешь самым первым делом с этим датасетом и почему.

    «Сначала посмотрю базовую информацию по датасету (.info(), .describe()), 
    проверю баланс классов в целевой переменной churn, и посмотрю на распределение признаков. 
    Это поможет понять, есть ли дисбаланс, пропуски или аномалии, и нужно ли их обрабатывать.»
"""
data.info()
data.describe()
print(f"Distribution of features: {data['churn'].value_counts(normalize=True)}")

print("CORRELATIONS")
print(data.corr(numeric_only=True))
"""
    Значение столбца has_debt сильно коррелирует с churn (1.0), поэтому от него придется избавиться
"""

#----------------------------------- 2/DATA SPLIT -----------------------------------------------
"""
Вопрос: как бы ты разделил данные на train и test (или validation)?

«Если данные статичные, я разделю случайно с учётом стратификации по таргету, 
чтобы баланс классов сохранился. Если данные временные, то сделаю временной split: 
train = прошлые периоды, test = будущее, чтобы избежать утечки информации.»
"""

X: DataFrame = data.drop(columns=["has_debt", "churn"])
y: Series = data['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, # доля тестовой выборки 20%
    stratify=y, # Делает так, что в train и test будет одинаковая пропорция классов, как в исходных данных (например, 40% ушли, 60% остались). Без этого при сильном дисбалансе в тесте может оказаться слишком мало положительных примеров.
    random_state=42) # Фикс случайное разбиение

"""
«Я использую train_test_split с параметрами test_size=0.2, stratify=y, 
чтобы сохранить баланс классов, и фиксирую random_state для воспроизводимости результатов.»
"""

#-------------------------------------3/Baseline Logistic Regression---------------------------

# fitting
model = LogisticRegression()
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

for real, pred in zip(y_test.values, y_pred):
    print(f"{real} : {pred}")

#-------------------------------------------------------------------------------------------------