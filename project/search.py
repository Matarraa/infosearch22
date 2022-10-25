import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import matrix_bm25
import matrix_bert
import matrix_tfidf

def find_answers_tfidf(query, vectorizer, index, answers):
    query_vector = matrix_tfidf.query_index(vectorizer, query)
    cos_similarities = np.squeeze(cosine_similarity(query_vector, index))
    ind = np.argsort(cos_similarities, axis=0)
    results = np.array(answers)[ind][::-1].squeeze()
    return results

def find_answers_bm25(query, count_vectorizer, index, answers):
    query_vector = matrix_bm25.query_indexation(query, count_vectorizer)
    bm25 = index.dot(query_vector.T)
    ind = np.argsort(bm25.toarray(), axis=0)
    results = np.array(answers)[ind][::-1].squeeze()
    return results

def find_answers_bert(query, tokenizer, model, index, answers):
    query_vector = matrix_bert.query_bert(tokenizer, model, query)
    cos_sims = np.squeeze(cosine_similarity(query_vector, index))
    ind = np.argsort(cos_sims, axis=0)
    results = np.array(answers)[ind][::-1].squeeze()
    return results

def main():
    print('Поиск ответов...')
    docs = find_answers(query, matrix, list(corpus.values()), method)
    print(*docs[:30], sep='\n\n')

if __name__ == '__main__':
    main()
