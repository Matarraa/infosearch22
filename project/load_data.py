import pickle
from transformers import AutoTokenizer, AutoModel, models
import torch
import preprocessing

def load_transformers_models():
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    return tokenizer, model

def load_pickle(filename):
    with open(filename, 'rb') as bf:
        pickle_object = pickle.load(bf)
    return pickle_object

def load_embedings_bert(filename):
    corpus_embeddings = torch.load(filename)
    return corpus_embeddings

def load_data():
    corpus = preprocessing.make_qa_dict('data.jsonl')
    #tfidf
    tfidf_index = load_pickle('tfidf_index.pickle')
    tfidf_vectorizer = load_pickle('tfidf_vectorizer.pickle')
    #bm25
    bm25_index = load_pickle('bm25_index.pickle')
    bm25_vectorizer = load_pickle('bm25_count_vectorizer.pickle')
    #bert
    bert_matrix = load_embedings_bert('tensor.pt')
    bert_tokenizer, bert_model = load_transformers_models()

    models = {
        'tfidf_index': tfidf_index,
        'tfidf_vectorizer': tfidf_vectorizer,
        'bm25_index': bm25_index,
        'bm25_vectorizer': bm25_vectorizer,
        'bert_tokenizer': bert_tokenizer,
        'bert_model': bert_model,
        'bert_index': bert_matrix
    }
    return models, corpus

models, corpus = load_data()