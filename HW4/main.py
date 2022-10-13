import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModel
import preprocessing
import matrix_bm25
import matrix_bert
import torch

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
        BERT = corpus.dot(query_vector.T)
        ind = np.argsort(BERT.toarray(), axis=0)
        results = np.array(answers)[ind][::-1].squeeze()
    return results

def load_corpus(filename):
    corpus_embeddings = torch.load('tensor.pt')
    return corpus_embeddings

def main():
    argument_parser = create_parser()
    args = argument_parser.parse_args()

    query = args.query
    method = args.method

    print('Предобработка корпуса...')
    corpus = preprocessing.make_qa_dict(args.path)
    print(len(corpus))

    if method == 'bm25':
        matrix = matrix_bm25.indexation_bm25(list(corpus.keys()))
    elif method == 'bert':
        print('Считаю матрицу BERT...')
        matrix = matrix_bert.indexation_bert(list(corpus.keys()), tokenizer, model)

    print('Идет поиск ответов...')
    docs = find_answers(query, matrix, list(corpus.values()), method)
    print(*docs[:30], sep='\n\n')

if __name__ == '__main__':
    main()
