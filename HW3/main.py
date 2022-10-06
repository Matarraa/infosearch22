import numpy as np
import argparse
import preprocessing
import indexation

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', type=str, help='the path to directory')
    parser.add_argument(
        '--query', type=str, help='query for search')
    return parser

def find_answers(query, corpus, answers):
    query_vector = indexation.query_indexation(query)
    bm25 = corpus.dot(query_vector.T)
    ind = np.argsort(bm25.toarray(), axis=0)
    return np.array(answers)[ind][::-1].squeeze()

def main():
    argument_parser = create_parser()
    args = argument_parser.parse_args()

    corpus = preprocessing.make_qa_dict(args.path)
    matrix = indexation.indexation(list(corpus.keys()))
    query = args.query
    docs = find_answers(query, matrix, list(corpus.values()))
    print(*docs[:30], sep='\n\n')

if __name__ == '__main__':
    main()




