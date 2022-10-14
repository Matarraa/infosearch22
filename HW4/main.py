import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import preprocessing
import matrix_bm25
import matrix_bert
import torch
import os

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', type=str, help='the path to directory')
    parser.add_argument(
        '--query', type=str, help='query for search')
    parser.add_argument(
        '--method', type=str, help='method: bm25/bert')
    return parser

def find_answers(query, corpus, answers, method):
    if method == 'bm25':
        query_vector = matrix_bm25.query_indexation(query)
        bm25 = corpus.dot(query_vector.T)
        ind = np.argsort(bm25.toarray(), axis=0)
        results = np.array(answers)[ind][::-1].squeeze()
    elif method == 'bert':
        query_vector = matrix_bert.query_bert(tokenizer, model, query)
        cos_sims = np.squeeze(cosine_similarity(query_vector, corpus))
        ind = np.argsort(cos_sims, axis=0)
        results = np.array(answers)[ind][::-1].squeeze()
    return results

def load_corpus(filename):
    corpus_embeddings = torch.load(filename)
    return corpus_embeddings

def main():
    argument_parser = create_parser()
    args = argument_parser.parse_args()

    query = args.query
    method = args.method

    print('Предобработка корпуса...')
    corpus = preprocessing.make_qa_dict(args.path)

    if method == 'bm25':
        matrix = matrix_bm25.indexation_bm25(list(corpus.keys()))
    elif method == 'bert':
        if 'tensor.pt' not in os.listdir():
            print('Индексация корпуса...')
            matrix = matrix_bert.indexation_bert(list(corpus.keys()), tokenizer, model)
        else:
            matrix = load_corpus('tensor.pt')

    print('Поиск ответов...')
    docs = find_answers(query, matrix, list(corpus.values()), method)
    print(*docs[:30], sep='\n\n')

if __name__ == '__main__':
    main()
