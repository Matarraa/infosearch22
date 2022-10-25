from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing

def inverted_index(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def query_index(vectorizer, query):
    query_tokens = preprocessing.text_preprocessing(query)
    query_vector = vectorizer.transform([' '.join(query_tokens)]).toarray()
    return query_vector
