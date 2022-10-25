import search

def main(method, query, models, corpus):
    if query == '':
        return ''
    if method == 'TF-IDF':
        results = search.find_answers_tfidf(query, models['tfidf_vectorizer'], models['tfidf_index'], list(corpus.values()))
    if method == 'Okapi BM25':
        results = search.find_answers_bm25(query, models['bm25_vectorizer'], models['bm25_index'], list(corpus.values()))
    if method == 'BERT':
        results = search.find_answers_bert(query, models['bert_tokenizer'], models['bert_model'], models['bert_index'], list(corpus.values()))
    return results