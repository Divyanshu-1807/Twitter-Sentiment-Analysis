import streamlit as st
import pandas as pd
import pickle
from nltk.stem import PorterStemmer
import re

tfid=pickle.load(open('tfid.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

ps=PorterStemmer()

def action(text):
    text=text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub('\s+', ' ', text)

    y=[]
    for word in text.split():
        if word not in stop_words :
          y.append(word)

    text=y[:]
    y.clear()
    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

st.title("TWITTER SENTIMENT ANALYSIS")
text=st.text_input("",placeholder="Enter tweet")
text_input=action(text)
input_text=tfid.transform([text_input])
result=model.predict(input_text)

if st.button('Predict',type="primary",use_container_width=True): 
    if(result==0):
        st.header("Negative :white_frowning_face:")
        st.toast('Moye Moye')
        #st.snow()
        audio_file = open('moye.wav','rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
    else:
        st.header("Positive :smile:")        
        st.toast('Yeah!!',icon='🎉')
        #st.balloons()
        audio_file = open('hoye.wav','rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

st.image('twt1.jpg', caption='Positive Tweets')
st.image('twt1.webp', caption='Negative Tweets')
st.image('twt3.jpg', caption='Positive Tweets')

