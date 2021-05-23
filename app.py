import joblib
from flask import Flask, request, render_template

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

app = Flask(__name__)

#Loading the model and tfidf vectorizer from disk
vector = joblib.load('netflix_mnb_vector.pkl')
model = joblib.load('netflix_sentiment_MNB.pkl')

def MNB(review):
    vectorizer = vector.transform([review])
    my_pred = model.predict(vectorizer)

    if my_pred==1:
        return('The review is Positive')
    else:
        return("The review is Negative")
    

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/FinalReview', methods = ['POST'])
def FinalReview():
    review = request.form['Review']
    result = MNB(review)

    return render_template('index.html',finalreview_text = result)

if __name__=="__main__":
    app.run(debug=True)