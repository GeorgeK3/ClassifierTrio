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

NAIVE BAYES

'''
#####################################################################################################################
class NaiveBayes:

    def calculate(x_train,y_train,x_test):
            
            predictions=[]
            
            #P(C=1)
            p1=np.mean(y_train.flatten() == 1)

            #P(C=0)
            p0=np.mean(y_train.flatten() == 0)
            

            num_features = len(x_train[0])  # Number of features in the dataset
            num_train_samples = len(y_train)

            for test in x_test:
 
                #P(Xi=xi|C=c)
                numerator=[[1]*num_features,[1]*num_features ] #LaPlace

                denominator=[2,2]  #LaPlace

                for i in range(num_train_samples):
                    denominator[y_train[i]] +=1
                    for j in range(num_features):
                        if x_train[i][j]==test[j]:
                            numerator[y_train[i]][j]+=1
                
                #Π(P(Xi=xi|C=c))
                numerator = np.array(numerator)
                denominator = np.array(denominator)
                quotient= np.prod(numerator / denominator[:, np.newaxis], axis=1)

                #P(C=0|X)
                p0_x = p0 * quotient[0]
                #P(C=1|X)
                p1_x = p1 * quotient[1]

                if p1_x > p0_x:
                    predictions.append(1)
                else:
                    predictions.append(0)

            return predictions
            
#####################################################################################################################
'''

TRYING IT OUT

'''
#####################################################################################################################
predictions = NaiveBayes.calculate(X_train.toarray(),Y_train,X_test.toarray())

accurancy = np.mean(predictions==Y_test)

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
