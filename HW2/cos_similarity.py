from sklearn.metrics.pairwise import cosine_similarity
import preprocessing

def query_index(vectorizer, query):
    query_tokens = preprocessing.text_preprocessing(query)
    query_vector = vectorizer.transform([' '.join(query_tokens)]).toarray()
    return query_vector

def count_similatiry(index, query_vector):
    cos_similarities = cosine_similarity(query_vector, index)
    return cos_similarities

