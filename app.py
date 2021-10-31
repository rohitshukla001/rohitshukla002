from flask import Flask, render_template,request
from nltk import corpus
#based level imports for data science work
import pandas as pd
import numpy as np 
import re,string
import joblib
import pickle

#NLP Libs
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

def text_cleaning(data):
    corpus=[]
    for i in range(0,len(data)):
        clean_data=re.sub(r'\W',' ',str(data[i]))
        clean_data=clean_data.lower()
        clean_data=re.sub(r'\d+'," ",clean_data)
        clean_data=re.sub(r"[^a-zA-Z]",' ',clean_data)
        clean_data=re.sub(r'\s+',' ',clean_data)
        corpus.append(clean_data)
    return corpus

# setting up flask name........
app=Flask(__name__)


#loading pkl files.............
model = joblib.load("model.pkl")
vector =joblib.load("vector.pkl")

#creating first app........
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        messagen = request.form['message']
        data=[messagen]
        clean_data=text_cleaning(data)
        vect_data=vector.transform(clean_data)
        prediction=model.predict(vect_data)
        if prediction == [0]:
            return render_template('index.html', prediction= [0])
        else:
            return render_template('index.html', prediction=[1])
if __name__ == "__main__":
    app.run(debug=True)