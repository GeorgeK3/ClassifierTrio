from statistics import mode
import numpy as np
import math
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

keep_m_words=60000
skip_n_first=20

(X_train_imdb, Y_train), (X_test_imdb, Y_test) = tf.keras.datasets.imdb.load_data('imdb.npz',keep_m_words,skip_n_first)

word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'
X_train_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in X_train_imdb])
X_test_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in X_test_imdb])


train_percentage = input("Please enter the percentage of data for the training(for exampe 0.3):  ")
train_percentage = float(train_percentage)  
if train_percentage>1 or train_percentage<=0 :
    print("The percentage you entered is not valid")
    exit()
    
test_percentage = 0.4

X_train_imdb, _, Y_train, _ = train_test_split(X_train_imdb, Y_train, train_size=train_percentage)
X_test_imdb, _, Y_test, _ = train_test_split(X_test_imdb, Y_test, test_size=test_percentage)

binary_vectorizer = CountVectorizer(binary=True, min_df=100)
X_train = binary_vectorizer.fit_transform(X_train_imdb)
X_test = binary_vectorizer.transform(X_test_imdb)
#####################################################################################################################

model = LogisticRegression()
model.fit(X_train.toarray(),Y_train)

y_pred = model.predict(X_test.toarray())

accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy = "+str(accuracy))

precision = precision_score(Y_test, y_pred)
print("Precision = "+str(precision))

recall = recall_score(Y_test, y_pred)
print("Recall = "+str(recall))

f1 = f1_score(Y_test, y_pred)
print("F1 = "+str(f1))
