from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
import preprocessing

def indexation_bm25(corpus, k=2, b=0.75):
    """Возвращает матрицу bm25"""
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    x_count = count_vectorizer.fit_transform(corpus)
    x_tf = x_count
    x_idf = tfidf_vectorizer.fit_transform(corpus)
    idf = tfidf_vectorizer.idf_
    len_d = x_count.sum(axis=1)
    avdl = len_d.mean()
    B_1 = k * (1 - b + b * len_d / avdl)
    matrix = sparse.lil_matrix(x_tf.shape)
    for i, j in zip(*x_tf.nonzero()):
        matrix[i, j] = (x_tf[i, j] * (k + 1) * idf[j])/(x_tf[i, j] + B_1[i])
    return matrix.tocsr(), count_vectorizer

def query_indexation(query, count_vectorizer):
    """Преобразовывает запрос в вектор"""
    query_lemmas = preprocessing.text_preprocessing(query)
    query_vector = count_vectorizer.transform([' '.join(query_lemmas)])
    return query_vector





