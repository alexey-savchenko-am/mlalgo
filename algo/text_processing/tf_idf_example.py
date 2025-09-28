from typing import List
from tf_idf import TfidfVectorizer

def main():
    
    docs: List[str] = [
        "I love this product!!",
        "Great value for money, love this.",
        "Cheap and useful, thank you",
        "Absolutely hate this stuff. Don't buy it.",
        "Hate, hate, hate!!!"
    ]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)

    print("Result Values")
    print(tfidf_matrix)



if __name__ == "__main__":
    main()