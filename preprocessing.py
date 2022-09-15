from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
m = Mystem()

def text_preprocessing(text):
    tokens = [word.strip(punctuation) for word in word_tokenize(text)]
    words = [word.lower() for word in tokens]
    sw = stopwords.words('russian')
    filtered_words = [word for word in words if word not in sw]
    lemmas = m.lemmatize(' '.join(filtered_words))
    clean_words = [word for word in lemmas if word.isalpha() or word.isdigit()]
    return(clean_words)

def inverted_index (corpus_dict):
    corpus_dict = [' '.join(list_lemmas) for list_lemmas in list(corpus_dict.values())]
    vectorizer = CountVectorizer(analyzer='word')
    X = vectorizer.fit_transform(corpus_dict)
    return vectorizer, X


def index_dict (corpus_dict):
    index_dict = defaultdict(list)
    for filename, lemmas in corpus_dict.items():
        counter = Counter(lemmas)
        for lemma in counter:
            index_dict[lemma].append([filename, counter[lemma]])
    return index_dict