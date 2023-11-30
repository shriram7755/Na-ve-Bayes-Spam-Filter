# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 21:58:19 2023

@author: SHRI
"""

import pandas as pd
import numpy as np

# read csv file
df = pd.read_csv('c:/2-dataset/spam.csv')
df.head()

df.Category.value_counts()

#create one more column spam ie 0 or 1 using category column
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df['spam']

df.shape
# (5572, 3)

#############################

#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.2)

#check shape of train and test data of X
X_train.shape
X_test.shape
#check type of above train and test data of X
print(type(X_train))
print(type(X_test))

#############################

#creating bag of words using count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()

X_train_cv = v.fit_transform(X_train.values)
X_train_cv
X_train_cv.shape
# (4457, 7719)

##############################

#train naive bayes model
from sklearn.naive_bayes import MultinomialNB
#initailization
model = MultinomialNB()
#train model
model.fit(X_train_cv, y_train)

###############################
#create BOW using countvectorizer
X_test_cv = v.transform(X_test)
X_test_cv.shape
###############################

#evaluate performance
from sklearn.metrics import classification_report
y_pred = model.predict(X_test_cv)
print(classification_report(y_test, y_pred))

#              precision    recall  f1-score   support
#
#           0       0.99      1.00      0.99       969
#           1       0.99      0.93      0.96       146
#
#    accuracy                           0.99      1115
#   macro avg       0.99      0.96      0.98      1115
#weighted avg       0.99      0.99      0.99      1115

# this is an overfit model since the accuracy is more than 90% ie 99%