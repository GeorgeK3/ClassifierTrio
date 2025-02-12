from statistics import mode
import numpy as np
import math
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.callbacks import History
import keras

#####################################################################################################################
'''

DATA 

'''
#####################################################################################################################
keep_m_words=60000
skip_n_first=20

(X_train_imdb, Y_train), (X_test_imdb, Y_test) = tf.keras.datasets.imdb.load_data('imdb.npz',keep_m_words,skip_n_first)
X_train_imdb = pad_sequences(X_train_imdb)
X_test_imdb = pad_sequences(X_test_imdb)
train_percentage = input("Please enter the percentage of data for the training(for exampe 0.3):  ")
train_percentage = float(train_percentage)  

if train_percentage>1 or train_percentage<=0 :
    print("The percentage you entered is not valid")
    exit()

test_percentage = 0.4

X_train_imdb, _, Y_train, _ = train_test_split(X_train_imdb, Y_train, train_size=train_percentage)
X_test_imdb, _, Y_test, _ = train_test_split(X_test_imdb, Y_test, test_size=test_percentage)

#####################################################################################################################
'''

RNN 

'''
#####################################################################################################################

model = Sequential()

model.add(keras.layers.Embedding(keep_m_words,32))

model.add(SimpleRNN(32))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = History()

model.fit(X_train_imdb, Y_train, epochs=10,validation_split=0.2,callbacks=[history])

#####################################################################################################################

y_pred = model.predict(X_test_imdb)
y_pred = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy = "+str(accuracy))

precision = precision_score(Y_test, y_pred)
print("Precision = "+str(precision))

recall = recall_score(Y_test, y_pred)
print("Recall = "+str(recall))

f1 = f1_score(Y_test, y_pred)
print("F1 = "+str(f1))


train_losses = history.history['loss']
val_losses = history.history['val_loss']

print("Training Losses:", train_losses)
print("Validation Losses:", val_losses)