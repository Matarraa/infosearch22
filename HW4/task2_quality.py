import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import json
m = Mystem()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse

count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', type=str, help='the path to directory')
    return parser

def text_preprocessing(text):
    """Предобработка текста: лемматизация, удаление пунктуации и стоп-слов"""
    tokens = [word.strip(punctuation) for word in word_tokenize(text)]
    words = [word.lower() for word in tokens]
    sw = stopwords.words('russian')
    filtered_words = [word for word in words if word not in sw]
    lemmas = m.lemmatize(' '.join(filtered_words))
    clean_words = [word for word in lemmas if word.isalpha() or word.isdigit()]
    return(clean_words)

def make_qa_dict_lemmas (filename):
    """Возвращает словарь: ключи - предобработанные вопросы, значения - предобработанные ответы"""
    with open(filename, 'r') as f:
        corpus = list(f)[:53000] # увеличила число вопросов, т.к. около 1000 удалились из-за отсутсвия ответов
    qa_dict = {}
    for element in corpus:
        if json.loads(element)['question'] != '':
            ans_value = 0
            text = ''
            for answer in json.loads(element)['answers']:
                value = 0 if answer['author_rating']['value'] == '' else int(answer['author_rating']['value'])
                if value > ans_value and answer['text'] != '':
                    ans_value = value
                    text = answer['text']
            if text != '':
                question = text_preprocessing(json.loads(element)['question'])
                ans_lemmas = text_preprocessing(text)
                qa_dict[' '.join(question)] = ' '.join(ans_lemmas)
    return qa_dict

def indexation_bm25(corpus, k=2, b=0.75):
    """Возвращает матрицу bm25"""
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
    return matrix.tocsr()

def get_score(answers_matrix, questions_matrix):
    sim = answers_matrix.dot(questions_matrix.T)
    counter = 0
    for i in range(sim.shape[0]):
        id_sort = np.argsort(sim[i], axis=0)[::-1]
        if i in id_sort[:5]:
            counter += 1
    return counter / questions_matrix.shape[0]

def load_corpus(filename):
    corpus_embeddings = torch.load(filename)
    return corpus_embeddings

def indexation_bert(corpus, tokenizer, model, filename):
    """ Возвращает матрицу bert """
    batch_size = 100
    tensors = tuple()
    for i in tqdm(range(0, len(corpus), batch_size)):
        texts = corpus[i: (i + batch_size)]
        encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        tensors += (batch_embeddings,)
    bert_matrix = normalize(torch.vstack(tensors))
    torch.save(bert_matrix, filename)
    return bert_matrix

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def main():
    argument_parser = create_parser()
    args = argument_parser.parse_args()

    print('Предобработка корпуса...')
    corpus = make_qa_dict_lemmas(args.path)

    # bm25
    a_matrix_bm25 = indexation_bm25(list(corpus.values()))
    q_matrix_bm25 = count_vectorizer.transform(list(corpus.keys()))
    bm25_res = get_score(a_matrix_bm25, q_matrix_bm25)

    # bert
    filename_q = 'tensor.pt'
    filename_a = 'tensor_ans.pt'
    if filename_q not in os.listdir():
        print('Индексация вопросов...')
        q_matrix_bert = indexation_bert(list(corpus.keys()), tokenizer, model, filename_q)
    else:
        q_matrix_bert = load_corpus(filename_q)

    if filename_a not in os.listdir():
        print('Индексация ответов...')
        a_matrix_bert = indexation_bert(list(corpus.keys()), tokenizer, model, filename_a)
    else:
        a_matrix_bert = load_corpus(filename_q)
    bert_res = get_score(a_matrix_bert, q_matrix_bert)

    print('Качество bm25: ', bm25_res)
    print('Качество bert: ', bert_res)


if __name__ == '__main__':
    main()
