#import all libraries

import pandas as pd
import numpy as np
import string
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

sw = stopwords.words('english')
wordnet = WordNetLemmatizer()

#Read the positive data

pos_rev = pd.read_csv("pos.txt", sep = '\n', header = None, encoding = 'Latin-1')

#Add a column

pos_rev['mood'] = 1.0
pos_rev = pos_rev.rename(columns = {0 :'review'})

#read negative data

neg_rev = pd.read_csv("negative.txt", sep = '\n', header = None, encoding = 'Latin-1')

#Add a column

neg_rev['mood'] = 0.0
neg_rev = neg_rev.rename(columns = {0 :'review'})


#Data preprocessing and cleaning the data

# 1.convert all text to lower
# 2.remove all spaces
# 3.remove all punctuations
# 4.remove all stopwords
# 5.Lemmatization

pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply(lambda x : x.lower())
pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply(lambda x : re.sub(r"@\S", " ", x))
pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply\
(lambda x : x.translate(str.maketrans(dict.fromkeys(string.punctuation))))
pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply\
(lambda x : " ".join([wordnet.lemmatize(word, pos = 'v') for word in x.split() if word not in (sw)]))

#concatenating the postive and negative reviews

com_rev = pd.concat([pos_rev, neg_rev], axis = 0).reset_index()

#train test split
X_train, X_test, y_train, y_test = train_test_split\
(com_rev['review'].values, com_rev['mood'].values, test_size = 0.2, random_state = 102)

train_data = pd.DataFrame({'review': X_train, 'mood': y_train})
test_data = pd.DataFrame({'review': X_test, 'mood': y_test})


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data['review'])
test_vectors = vectorizer.transform(test_data['review'])

MNB = MultinomialNB()
MNB.fit(train_vectors, y_train)
y_pred = MNB.predict(test_vectors)


#saving the model

model_file_name = 'netflix_sentiment_MNB.pkl'
vectorizer_file_name = 'netflix_mnb_vector.pkl'
joblib.dump(MNB, model_file_name)
joblib.dump(vectorizer, vectorizer_file_name)

