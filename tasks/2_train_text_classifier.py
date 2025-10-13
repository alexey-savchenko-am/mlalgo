import pandas as pd
from utils.dataset_loader import DatasetLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


#------------------- DATASET PREPARATION --------------------------
loader = DatasetLoader("./assets/topics.json")

data = loader.load()

loader.analyze_by("label")

data = data.dropna() #remove NaN strings
data = data[data["text"] != ""] #keep only non empty rows

duplicates = data["text"].duplicated().sum()
print(f"\n📌 Number of duplicates: {duplicates}")

data = data.drop_duplicates()

#------------------- TRAIN&TEST SPLIT --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2, # 20% test selection
    random_state=42,
    stratify=data["label"] # keep class ballance
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

#------------------- TRANSFORM TEXT --------------------------

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#------------------- FIT THE MODEL --------------------------

model = LogisticRegression(max_iter=100_000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

#------------------- REPORT  --------------------------
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

#------------------- CUSTOM   --------------------------

topics = [
    "We love playing football",
    "Listening to music is a great way to chill out",
    "Computers are the great source of information",
]
vectorized_topics = vectorizer.transform(topics)
y_pred = model.predict(vectorized_topics)

for topic, label in zip(topics, y_pred):
    print(f"{topic}: {label}")
