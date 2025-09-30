import pandas as pd
from utils.dataset_loader import DatasetLoader
from pathlib import Path
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt


def main():

    loader = DatasetLoader("./assets/topics.json")

    data = loader.load()
    #print(f"Distribution of features: {data['label'].value_counts(normalize=True)}")
 
    words = list(chain.from_iterable(data['text'].str.lower().str.split()))
    #print(Counter(words).most_common(20))


    # Check if there are empty texts
    #print(data.isnull().sum())

    empty_strings = (data['text'] == '').sum()
    #print("Empty strings in 'text':", empty_strings)

    data['label'].value_counts().plot(kind='bar')
    plt.show()

if __name__ == "__main__":
    main()