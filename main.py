import os
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer

import preprocessing

def making_corpus ():
    # создаем словарь с текстами
    corpus = {}
    for root, _, files in os.walk('friends-data'):
        for name in files:
            if name[0] != '.':
                path = os.path.join(root, name)
                with open(path, 'r', encoding='utf-8') as f:
                    corpus[name] = f.read()
    # проходим по текстам и обрабатываем их
    for filename, text in corpus.items():
        corpus[filename] = preprocessing.text_preprocessing(text)
    return corpus

def dict_statistics (index_dict):
    num_files = []
    for files in index_dict.values():
        n = 0
        for file in files:
           n += file[1]
        num_files.append(n)
    min_freq = min(num_files)
    max_freq = max(num_files)
    less_frequent = []
    most_frequent = []
    in_all_files = []
    names_dict = Counter()
    for word, files in index_dict.items():
        if len(files) == 165:
            in_all_files.append(word)  # слова, которые есть во всех документах
        freq = 0
        for file in files:
            freq += file[1]
            if  freq == min_freq:
                less_frequent.append(word)  # самые нечастотные слова
            if freq == max_freq:
                most_frequent.append(word)  # самые частотные слова
            for character, names in characters.items():
                if word.capitalize() in names:
                    names_dict[character] += freq  # частота встречаемости героев
    return most_frequent, less_frequent, in_all_files, names_dict.most_common()[0][0]

def matrix_statistics (vectorizer):
    # самое частотное слово
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    final_matrix = np.array([np.array(vectorizer.get_feature_names()), matrix_freq])
    # самое редкое слово
    # слова, которые есть во всех документах
    # самый популярный герой
    return (final_matrix, matrix_freq)

characters = {'Моника':['Моника', 'Мон'], 'Рэйчел':['Pэйчел', 'Рейч'],
              'Чендлер':['Чендлер', 'Чэндлер', 'Чен'], 'Фиби':['Фиби', 'Фибс'],
              'Росс':['Росс'], 'Джоуи':['Джоуи', 'Джои', 'Джо']}

corpus = making_corpus()
index_dict = preprocessing.index_dict (corpus)
vectorizer, X = preprocessing.inverted_index(corpus)

most_frequent, less_frequent, in_all_files, freq_character = dict_statistics (index_dict)
print('По словарю:\n\n', 'Самые частотные слова: ',', '.join(most_frequent), '\n',
      'Самые нечастотные слова: ', ', '.join(less_frequent), '\n',
      'Слова, которые втречаются во всех документах: ', ', '.join(in_all_files), '\n',
      'Самый статистически популярный герой: ', freq_character)

