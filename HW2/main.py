import os
import numpy as np
import argparse
import preprocessing
import cos_similarity

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', type=str, help='the path to directory')
    parser.add_argument(
        '--query', type=str, help='query for search')
    return parser

def making_corpus(dir_path):
    # создаем словарь с текстами
    corpus = {}
    for root, _, files in os.walk(dir_path):
        for name in files:
            if name[0] != '.':
                path = os.path.join(root, name)
                with open(path, 'r', encoding='utf-8') as f:
                    corpus[name] = f.read()
    # проходим по текстам и обрабатываем их
    for filename, text in corpus.items():
        corpus[filename] = preprocessing.text_preprocessing(text)
    return corpus

def main():
    argument_parser = create_parser()

    args = argument_parser.parse_args()

    corpus = making_corpus(args.path)
    vectorizer, X = preprocessing.inverted_index(corpus)

    query = args.query
    query_vector = cos_similarity.query_index(vectorizer, query)
    cos_similarities = cos_similarity.count_similatiry(X, query_vector)
    similarity_dict = {filename: similarity for filename, similarity in zip(list(corpus), list(cos_similarities[0]))}
    for key in sorted(similarity_dict, key=similarity_dict.get, reverse=True):
        print(key)

if __name__ == '__main__':
    main()

