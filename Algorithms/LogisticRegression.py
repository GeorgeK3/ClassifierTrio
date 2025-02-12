from statistics import mode
import numpy as np
import math
import random
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#####################################################################################################################
'''

DATA 

'''
#####################################################################################################################
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
'''

Logistic Regression

'''
#####################################################################################################################

def posibility(wx):
    return 1 / (1 + np.exp(-wx))

class LogisticRegression:

    def log_likelihood(X, y, w):
        p = posibility(X @ w)
        likelihood = y.T @ np.log(p) + (1 - y).T @ np.log(1 - p)
        return likelihood

    def gradient_ascent(X, y, w, h, l, iterations):
        log_likelihoods = []

        for _ in range(iterations):
            p =  posibility(X @ w)
            gradient = X.T @ (y - p)
            w = (1-2*l*h)*w + h*gradient
            likelihood =  LogisticRegression.log_likelihood(X, y, w)
            log_likelihoods.append(likelihood)

        return w, log_likelihoods
    

#####################################################################################################################
'''

TRYING IT OUT

'''
#####################################################################################################################

w_list=[0]*len(X_train.toarray()[0])
w = np.array(w_list)
#Set hyperparameters
h=0.05
iterations=1000
l=0.1

w, log_likelihoods = LogisticRegression.gradient_ascent(X_train.toarray(), Y_train, w, h, l,iterations)

predictions = np.round(posibility(X_test.toarray() @ w))

accurancy = np.mean(predictions == Y_test)
print("Tests used = " + str(len(Y_test)))
print ("Accurancy = "+ str(accurancy*100) + "%")
print("Samples used ~ "+str(train_percentage) )

hit=0
miss = 0
for i in range(len(Y_test)):
    if (Y_test[i]==1 and predictions[i]==1):
        hit +=1
    elif (Y_test[i]==1 and predictions[i]==0):
        miss+=1

#Precision for C=1
if(np.sum(predictions)>0):
    precision = hit/np.sum(predictions)
elif hit==0:
    precision=1

#Recall for C=1 
if((hit+miss)>0):
    recall=hit/(hit+miss)
elif hit==0:
    recall=1

#F1
F1=((1+1)*precision*recall)/(1*precision+recall)

print("F1 = "+str(F1)+"\nprecission = "+str(precision)+"\nrecall = "+str(recall))