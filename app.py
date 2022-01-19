import streamlit as st
from transformers import pipeline
import numpy as np
import pandas as pd
from datetime import datetime
import altair as alt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

st.title("Text Classification Pipeline")
st.write("""

    Find me at: [LinkedIn](https://www.linkedin.com/in/ayoub-nainia/) | [GitHub](https://github.com/nainiayoub) | [Twitter](https://twitter.com/nainia_ayoub)


""")
st.markdown("""
In this web app, we will leverage the most bast object in the
 `Transformers library` which is the `Pipeline`, 
 to work with text data.
""")

st.session_state.text = []
st.session_state.proba = []
st.session_state.label = []
st.session_state.data = []


with st.form(key='my_form'):
    text_input = st.text_input(label='Enter some text')
    submit_button = st.form_submit_button(label='Submit')


# Zero-shot classification
with st.expander("Specifying labels and classifier"):
    st.markdown('the `zero-shot-classification pipeline` is very powerful: it allows you to specify which labels to use for the classification, so you don’t have to rely on the labels of the pretrained model.')

    labels = []
    col1, col2, col3 = st.columns(3)
    with col1:
        label1 = st.text_input("Label 1")
        labels.append(label1)
    with col2:
        label2 = st.text_input("Label 2")
        labels.append(label2)
    with col3:
        label3 = st.text_input("Label 3")
        labels.append(label3)
     

with st.expander("Token Classification"):
  if text_input and labels:
    zero_shot = pipeline('zero-shot-classification') 
    st.write("__Text tokenization__")
    tokenizedText = word_tokenize(text_input)
    st.write(tokenizedText)

    st.write("__Remove stop words__")
    engStopWords = stopwords.words('english')
    tkndText = [i for i in tokenizedText if i.lower() not in engStopWords]
    st.write(tkndText)

    st.write("__Classifying tokens__")
    classified_label = []
    classified_proba = []
    tokens = []
    for i in tkndText:
      # tokens.append(i)
      classified_tkn = zero_shot(i, candidate_labels=labels)

      tkn_proba = np.max(classified_tkn['scores'])
      # classified_proba.append(tkn_proba)

      index = classified_tkn['scores'].index(tkn_proba)
      # classified_label.append(classified_label[index])
      st.write(f"TOKEN: {i} --- LABEL: {classified_tkn['labels'][index]} --- SCORE: {tkn_proba}")


with st.expander("Text Classification"):
  st.write('Classification result')
  if text_input and labels:
    if zero_shot:
      classified = zero_shot(text_input, candidate_labels=labels)
      # date
      datePrediction = datetime.now()
      # Score and index of label
      score = np.max(classified['scores'])
      index = classified['scores'].index(score)

    
      # st.write(classified['labels'])
      # st.write(classified['scores'])
      # st.write(type(classified['scores']))
      if st.button('Classify'):
        # Results
        col3, col4 = st.columns(2)
        with col3:
          st.success(f"Predicted context: {(classified['labels'][index]).upper()}")
        
        with col4:
          st.success(f"Confidence: {score * 100}%")
        proba_df = pd.DataFrame(classified['scores'])
        proba_df = proba_df.transpose()
        proba_df.columns = classified['labels']
        st.write(text_input)
        st.write(proba_df)
      # st.write(proba_df.T)

      # plotting probability
      # proba_df_clean = proba_df.T.reset_index()
      # proba_df_clean.columns = ["Labels", "probability"]

      # fig = alt.Chart(proba_df).mark_bar().encode(x='labels', y='probability', color='labels')
      # st.altair_chart(fig, use_container_width=True)

      # Plotting the classifications by score
      # st.write(classified['scores'])
      # st.write(classified['labels'])
      # score_df = pd.DataFrame(classified['scores'], columns=classified['labels'])
      # st.write(score_df)


      