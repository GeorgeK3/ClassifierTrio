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

ID3

'''
#####################################################################################################################

class Node:
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category


class ID3:
    def __init__(self, features):
        self.tree = None
        self.features = features
    
    def fit(self, x, y):
        '''
        creates the tree
        '''
        most_common = mode(y.flatten())
        self.tree = self.create_tree(x, y, features=np.arange(len(self.features)), category=most_common)
        return self.tree
    
    def create_tree(self, x_train, y_train, features, category,max_depth=6, current_depth=0, threshold=0.9):
        # check empty data
        if len(x_train) == 0 or current_depth == max_depth:
            return Node(checking_feature=None, is_leaf=True, category=category)  # decision node
        # check if the examples belonging in one category are above the threshold
        if np.mean(y_train.flatten() == 0)>threshold:
            return Node(checking_feature=None, is_leaf=True, category=0)
        elif np.mean(y_train.flatten() == 1)>threshold:
            return Node(checking_feature=None, is_leaf=True, category=1)
        if len(features) == 0:
            return Node(checking_feature=None, is_leaf=True, category=mode(y_train.flatten()))
        igs = list()
        for feat_index in features.flatten():
            igs.append(self.calculate_ig(y_train.flatten(), [example[feat_index] for example in x_train]))
        max_ig_idx = np.argmax(np.array(igs).flatten())
        m = mode(y_train.flatten())  # most common category 

        root = Node(checking_feature=max_ig_idx)

        # data subset with X = 0
        x_train_0 = x_train[x_train[:, max_ig_idx] == 0, :]
        y_train_0 = y_train[x_train[:, max_ig_idx] == 0].flatten()

        # data subset with X = 1
        x_train_1 = x_train[x_train[:, max_ig_idx] == 1, :]
        y_train_1 = y_train[x_train[:, max_ig_idx] == 1].flatten()

        new_features_indices = np.delete(features.flatten(), max_ig_idx)  # remove current feature

        root.left_child = self.create_tree(x_train=x_train_1, y_train=y_train_1, features=new_features_indices, 
                                           category=m,max_depth=max_depth, current_depth=current_depth + 1)  # go left for X = 1
        
        root.right_child = self.create_tree(x_train=x_train_0, y_train=y_train_0, features=new_features_indices,
                                            category=m,max_depth=max_depth, current_depth=current_depth + 1)  # go right for X = 0
        
        return root


    @staticmethod
    def calculate_ig(classes_vector, feature):
        classes = set(classes_vector)

        HC = 0
    
        for c in classes:
            PC = list(classes_vector).count(c) / len(classes_vector)  # P(C=c)
            HC += - PC * math.log(PC, 2)  # H(C)
            # print('Overall Entropy:', HC)  # entropy for C variable
            
        feature_values = set(feature)  # 0 or 1 in this example
        HC_feature = 0
        
        for value in feature_values:
            # pf --> P(X=x)
            pf = list(feature).count(value) / len(feature)  # count occurences of value 
            indices = [i for i in range(len(feature)) if feature[i] == value]  # rows (examples) that have X=x

            classes_of_feat = [classes_vector[i] for i in indices]  # category of examples listed in indices above
            for c in classes:
                # pcf --> P(C=c|X=x)
                pcf = classes_of_feat.count(c) / len(classes_of_feat)  # given X=x, count C
                if pcf != 0: 
                    # - P(X=x) * P(C=c|X=x) * log2(P(C=c|X=x))
                    temp_H = - pf * pcf * math.log(pcf, 2)
                    # sum for all values of C (class) and X (values of specific feature)
                    HC_feature += temp_H
        
        ig = HC - HC_feature
        return ig    

        

    def predict(self, x):
        predicted_classes = list()

        for unlabeled in x:  # for every example

            if isinstance(self,ID3):
                tmp = self.tree  # begin at root
            else:
                tmp = self

            while not tmp.is_leaf:
                if unlabeled.flatten()[tmp.checking_feature] == 1:
                    tmp = tmp.left_child
                else:
                    tmp = tmp.right_child
            
            predicted_classes.append(tmp.category)
        
        return np.array(predicted_classes)
    

#####################################################################################################################
'''

RANDOM FOREST

'''
#####################################################################################################################

class RandomForest:
    def __init__(self, n_trees=10, max_features=None):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, x, y,keep_samples=0.8):
        n_samples, n_features = x.shape

        for i in range(self.n_trees):
            # Bootstrap sampling with replacement for creating subsets of data
            indices = np.random.choice(n_samples, replace=True, size=(int(n_samples*keep_samples)))
            x_bootstrap = x[indices]
            y_bootstrap = y[indices]
           
            # Random feature selection if specified
            if self.max_features is not None:
                selected_features = np.random.choice(n_features, size=self.max_features, replace=False)
                x_bootstrap = x_bootstrap[:, selected_features]
            id3 = ID3(features=range(x_bootstrap.shape[1]))  # Initialize ID3 with features
            tree = id3.fit(x_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, x):
        predictions = []

        for tree in self.trees:

            tree_predictions = []
            
            tree_predictions.append( ID3.predict(tree,x))

            predictions.append(np.array(tree_predictions).flatten())

        # Aggregate predictions (e.g., take the majority vote for classification)
            
        aggregated_predictions = np.array(predictions).T
        final_predictions = [np.bincount(prediction_row).argmax() for prediction_row in aggregated_predictions]

        return np.array(final_predictions)


#####################################################################################################################
'''

TRYING IT OUT

'''
#####################################################################################################################
random_forest = RandomForest(n_trees=10)
random_forest.fit(X_train.toarray(), Y_train)
predictions = random_forest.predict(X_test.toarray())

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