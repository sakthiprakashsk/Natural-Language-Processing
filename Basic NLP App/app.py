from transformers import pipeline
import streamlit as st
import pandas as pd

st.title(" NLP APP")
menu = ["__ Select__", "Text Classification", "Text Summarization"]
input = st.sidebar.selectbox("Choice any one",menu)

if input == "__Select__":
    st.write(" This is a Natural Language Processing Based Web App that can do anything u can imagine with the Text.")

elif input == "Text Classification":
    st.header("Text Classification")
    input_text=st.text_input("Enter the text..")
    submit=st.button("Submit")
    if submit :
        classifier = pipeline('text-classification', model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        result=classifier(input_text)
        sentiment = result[0]['label']
        if sentiment== "POSITIVE":
            st.write("The text has a Positive Sentiment")
        elif sentiment== "NEGATIVE":
            st.write("The text has a Negative Sentiment")
        elif sentiment== "NEUTRAL":
            st.write("The text seems Neutral Sentiment")


elif input == "Text Summarization":
    st.header("Text Summarization")
    input_text=st.text_area("Enter the text..")
    submit=st.button("Submit")
    if submit:
        text_summirizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        result=text_summirizer(input_text)
        print(result)

        
