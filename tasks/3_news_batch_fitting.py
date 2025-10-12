from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from utils.dataset_loader import DatasetLoader

def clean_tag(tag: str) -> str:
    return tag.lower().strip().strip('.,?!\'"')

loader = DatasetLoader('./assets/news.json')

known_tags = set()
for batch in loader.load_batch(batch_size=100):
    for tags in batch["tags"]:
        cleaned = [clean_tag(t) for t in tags if t.strip()]
        known_tags.update(cleaned)

all_tags = sorted(list(known_tags))

mlb = MultiLabelBinarizer(classes=all_tags)
mlb.fit([[]])

vectorizer = HashingVectorizer(
    n_features=2**18,
    alternate_sign=False,
    ngram_range=(1, 2)
)

clfs = [SGDClassifier(loss='log_loss', max_iter=1000) for _ in range(len(all_tags))]
classes = np.array([0, 1])

for batch in loader.load_batch(batch_size=100):
    data: pd.DataFrame = batch[batch["content"] != ""]
    if data["content"].duplicated().sum() > 0:
        data = data.drop_duplicates(subset="content")

    data["tags"] = data["tags"].apply(lambda lst: [clean_tag(t) for t in lst if t.strip()])

    X = vectorizer.transform(data["content"])
    Y = mlb.transform(data["tags"])

    for i, clf in enumerate(clfs):
        clf.partial_fit(X, Y[:, i], classes=classes)

print("Model fitted!")

new_articles = [
    "Students in Gaza struggle to concentrate amid political tensions",
    "Israel and neighbors discuss strategies for regional stability",
    "Educational institutions adopt new technologies to maintain learning quality",
    "Commentators analyze international policy changes affecting trade and security",
    "Climate change threatens food supply in multiple countries",
    "Middle East and North Africa see rising humanitarian crises",
    "Academics call for more support for remote learning",
    "Global initiatives aim to reduce inequality and poverty worldwide"
]

X_new_vec = vectorizer.transform(new_articles)

proba_matrix = np.zeros((len(new_articles), len(all_tags)))

for i, clf in enumerate(clfs):
    proba_matrix[:, i] = clf.predict_proba(X_new_vec)[:, 1]

top_k = 4
predicted_tags = []

for probs in proba_matrix:
    top_indices = np.argsort(probs)[-top_k:][::-1]  # индексы 4 самых больших значений
    top_tags = [all_tags[i] for i in top_indices]
    predicted_tags.append(top_tags)
    
for article, tags in zip(new_articles, predicted_tags):
    print(f"\n📰 Article: {article}")
    print(f"🏷️ Predicted tags: {tags}")
