import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def query_bert(tokenizer, model, text):
    """ Индексируем запрос """
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_vector = mean_pooling(model_output, encoded_input['attention_mask'])
    return normalize(query_vector)

def indexation_bert(corpus, tokenizer, model):
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
    torch.save(bert_matrix, 'tensor.pt')
    return bert_matrix






