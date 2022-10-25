import time
import streamlit as st
from load_data import models, corpus
import main

st.set_page_config(layout="wide")

picture = open('man.png', 'rb')
pic_bytes = picture.read()
picture2 = open('Heart2.png', 'rb')
pic2_bytes = picture2.read()

title_cols = st.columns([2.2, 1])
title_cols[0].title('Поисковик с любовью')
title_cols[1].image(pic_bytes, output_format='PNG')
title_cols[0].subheader('Здесь ты можешь найти ответ на любой вопрос о делах любовных!')
title_cols[0].markdown('**Введи волнующий тебя вопрос и выбери один из методов поиска:**\n'
                   '- TF-IDF \n- Okapi BM25 \n- BERT\n')

query_col, method_col, button_col = st.columns([3, 1, 0.2])
method = method_col.selectbox(label='Метод', label_visibility='collapsed', options=['TF-IDF', 'Okapi BM25', 'BERT'])
query = query_col.text_input(label='Твой запрос', label_visibility='collapsed', placeholder='Твой запрос', key='query')
search_button = button_col.button(label='Поиск')

# search
if search_button:
    placeholder = st.empty()
    start_time = time.time()
    results = main.main(method, query, models, corpus)
    work_time = round((time.time() - start_time), 3)
    if isinstance(results, str) and results == '':
        placeholder.error('Пустой запрос!')
    else:
        time_placeholder = st.caption(f'Поиск был произведён за {work_time} секунд')
        for elem in results[:30]:
            with st.container():
                results_cols = st.columns([0.5, 7])
                results_cols[0].image(pic2_bytes, width=30)
                results_cols[1].write(elem)

