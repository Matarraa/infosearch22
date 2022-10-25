from transformers import AutoTokenizer, AutoModel
import preprocessing
import matrix_bm25
import matrix_bert
import matrix_tfidf
import torch
import os
import pickle

def load_transformers_models():
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    return tokenizer, model

def making_matrices(filename):
    print('Предобработка корпуса...')
    corpus = preprocessing.make_qa_dict(filename)

    #tfidf
    tfidf_vectorizer, tfidf_matrix = matrix_tfidf.inverted_index(list(corpus.keys()))
    with open('tfidf_index.pickle', 'wb') as file:
        pickle.dump(tfidf_matrix, file)
    with open('tfidf_vectorizer.pickle', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)
    print('TFIDF DONE')

    #bm25
    bm25_matrix, count_vectorizer = matrix_bm25.indexation_bm25(list(corpus.keys()))
    with open('bm25_index.pickle', 'wb') as file:
        pickle.dump(bm25_matrix, file)
    with open('bm25_count_vectorizer.pickle', 'wb') as file:
        pickle.dump(count_vectorizer, file)
    print('BM25 DONE')

    #bert
    bert_tokenizer, bert_model = load_transformers_models()
    if 'tensor.pt' not in os.listdir():
        print('Индексация корпуса...')
        bert_matrix = matrix_bert.indexation_bert(list(corpus.keys()), bert_tokenizer, bert_model)
    else:
        print('BERT DONE')

if __name__ == '__main__':
    making_matrices('data.jsonl')