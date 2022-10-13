from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import json
m = Mystem()

def text_preprocessing(text):
    """Предобработка текста: лемматизация, удаление пунктуации и стоп-слов"""
    tokens = [word.strip(punctuation) for word in word_tokenize(text)]
    words = [word.lower() for word in tokens]
    sw = stopwords.words('russian')
    filtered_words = [word for word in words if word not in sw]
    lemmas = m.lemmatize(' '.join(filtered_words))
    clean_words = [word for word in lemmas if word.isalpha() or word.isdigit()]
    return(clean_words)

def make_qa_dict (filename):
    """Возвращает словарь: ключи - предобработанные вопросы, значения - ответы"""
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
                qa_dict[' '.join(question)] = text
    return qa_dict




