import re
import numpy as np
from collections import Counter
from typing import List, Dict

class TfidfVectorizer:
    def __init__(self):
        tokenized: List[List[str]] = []
        # Vocabulary: list of unique words across all documents
        self.vocab: List[str] = []
        # Mapping from word -> index in the matrix
        self.word2idx: Dict[str, int] = {}
        # IDF values for words
        self.idf: np.ndarray | None = None
        # Total number of documents
        self.N: int = 0

    def fit(self, docs: List[str]) -> "TfidfVectorizer":
        """
        Build vocabulary and compute IDF values.
        :param docs: list of documents (strings)
        :return: self
        """
        self.N = len(docs)
        self.tokenized = [self.tokenize(doc) for doc in docs]
        
        #collect vocabulary (unique words)
        self.vocab = sorted(set(word for doc in self.tokenized for word in doc))
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}

        # Compute document rewquency (DF)
        df = np.zeros(len(self.vocab), dtype=int)

        for doc in self.tokenized:
            unique_words = set(doc)
            for word in unique_words:
                df[self.word2idx[word]] += 1

        self.idf = np.log(self.N / (1 + df))

        return self
    
    def transform(self) -> np.ndarray:
        """
        Transform documents into a TF-IDF matrix (NumPy ndarray).
        :param docs: list of documents (strings)
        :return: TF-IDF matrix (shape: [n_docs, n_features])
        """
        if self.idf is None:
            raise ValueError("Model is not fitted yet. Call fit() or fit_transform().")
        
        n_docs = self.N
        n_features = len(self.vocab)

        # Initialize result matrix
        tfidf_matrix = np.zeros((n_docs, n_features), dtype=float)

        for i, doc in enumerate(self.tokenized):
            word_counts = Counter(doc)
            doc_len = len(doc)

            for word, count in word_counts.items():
                if word in self.word2idx:
                    j = self.word2idx[word]
                    tf = count / doc_len if doc_len > 0 else 0.0
                    tfidf_matrix[i, j] = tf * self.idf[j]

        return tfidf_matrix
    
    def fit_transform(self, docs: List[str]) -> np.ndarray:
        """
        Fit on the given documents and return their TF-IDF matrix.
        """
        self.fit(docs)
        return self.transform()

    def get_feature_names(self) -> List[str]:
        """
        Return the list of words (features).
        """
        return self.vocab

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Simple regex-based tokenizer:
        - converts text to lowercase
        - extracts only alphanumeric word tokens
        """
        return re.findall(r"\b\w+\b", text.lower())