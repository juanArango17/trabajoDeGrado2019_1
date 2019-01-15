# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:43:20 2018

@author: user
"""

import db_info_extracton as db
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
#import pickle
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.svm import SVC

#from sklearn.pipeline import Pipeline

#%%Defining Tokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer

class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer(r'some_regular_expression')
        self.stemmer = LancasterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(token) 
                for token in self.tok.tokenize(doc)]
        
from sklearn.decomposition import TruncatedSVD

svd_model = TruncatedSVD(n_components=10000, 
                         algorithm='randomized',
                         n_iter=10, 
                         random_state=42)
        
#%% Defining TF-IDF structure

TFIDFvectorizer = TfidfVectorizer(use_idf=True,
                                  smooth_idf=True,
                                  encoding = 'latin2')

#%%Load database

all_files = []

all_files.append((db.BBC_Dataset_setup('/home/jparango/Documents/Trabajo_de_grado_2019_1/Dataset/bbc/business'),0))
all_files.append((db.BBC_Dataset_setup('/home/jparango/Documents/Trabajo_de_grado_2019_1/Dataset/bbc/entertainment'),1))
all_files.append((db.BBC_Dataset_setup('/home/jparango/Documents/Trabajo_de_grado_2019_1/Dataset/bbc/politics'),2))
all_files.append((db.BBC_Dataset_setup('/home/jparango/Documents/Trabajo_de_grado_2019_1/Dataset/bbc/tech'),3))
all_files.append((db.BBC_Dataset_setup('/home/jparango/Documents/Trabajo_de_grado_2019_1/Dataset/bbc/sport'),4))

#%%
documents = []
y = []
for i in range(len(all_files)):

    for j in range(len(all_files[i][0])):
        documents.append(all_files[i][0][j])
        y.append(i)
DB_size = len(documents)
print('tamaño de la DB: '+str(DB_size))

#%%Processing info
"""
svd_transformer = Pipeline([('tfidf', TFIDFvectorizer), 
                            ('svd', svd_model)])
svd_matrix = svd_transformer.fit_transform(documents)
"""

#%%

vectors = TFIDFvectorizer.fit_transform(documents)

indexes = [i for i in range(len(documents))]
random.shuffle(indexes)

X_random = np.copy(vectors.A)
y_random = np.copy(y)
for i in range(len(indexes)):
    X_random[i] = X_random[indexes[i]]
    y_random[i] = y_random[indexes[i]]

X_train, X_test, y_train, y_test = train_test_split(X_random, y_random, test_size=0.15, random_state=0)

scaler          = StandardScaler()
pca             = PCA(n_components = 0.95, svd_solver = 'full')
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train_reduced = pca.fit_transform(X_train)
X_test_reduced  = pca.transform(X_test)
print('Reduction of components: '+str(100*X_train_reduced.shape[1]/X_train.shape[1])+'%')

# %% SVM Training Algorithm

print('*********')
print('Running SVM Algorithm')
print('*********')

for c in list(np.logspace(-1,3,num=10)):
    svm_classifier = SVC(C = c, kernel = 'rbf')
    svm_classifier.fit(X_train_reduced, y_train)
    
    Predictions_train = svm_classifier.predict(X_train_reduced)
    Predictions_test  = svm_classifier.predict(X_test_reduced )
    
    accuracy_train = f1_score(Predictions_train,y_train, average='micro')
    accuracy_test  = f1_score(Predictions_test, y_test , average='micro')
    
    print('for C = '+str(c)+': ')

    print('   f1_score_train: '+str(accuracy_train))
    print('   f1_score_test : '+str(accuracy_test ))
    
    cm_train = confusion_matrix(y_train, Predictions_train)
    cm_test  = confusion_matrix(y_test , Predictions_test )
    
    print('   confusion matrix for train data')
    print(cm_train)
    print('   confusion matrix for test data ')
    print(cm_test )
    
    print('****************************************')
    print('****************************************')
    print('****************************************')
    
#%%
svm_classifier = SVC(C = 0.2782559402207124, kernel = 'rbf')
svm_classifier.fit(X_train_reduced, y_train)

Predictions_train = svm_classifier.predict(X_train_reduced)
Predictions_test  = svm_classifier.predict(X_test_reduced )

accuracy_train = f1_score(Predictions_train,y_train, average='micro')
accuracy_test  = f1_score(Predictions_test, y_test , average='micro')

#print('for C = '+str(c)+': ')

print('   f1_score_train: '+str(accuracy_train))
print('   f1_score_test : '+str(accuracy_test ))

cm_train = confusion_matrix(y_train, Predictions_train)
cm_test  = confusion_matrix(y_test , Predictions_test )

print('   confusion matrix for train data')
print(cm_train)
print('   confusion matrix for test data ')
print(cm_test )

print('****************************************')
print('****************************************')
print('****************************************')

#%% KNN Training Algorithm

print('*********')
print('Running KNN Algorithm')
print('*********')

for k in range(2,6+1):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train_reduced, y_train)
    
    Predictions_train = knn_classifier.predict(X_train_reduced)
    Predictions_test  = knn_classifier.predict(X_test_reduced )
    
    accuracy_train = f1_score(Predictions_train,y_train, average='micro')
    accuracy_test  = f1_score(Predictions_test, y_test , average='micro')
    
    print('for n_neighbors = '+str(k)+': ')
    
    print('   f1_score_train: '+str(accuracy_train))
    print('   f1_score_test : '+str(accuracy_test ))
    
    cm_train = confusion_matrix(y_train, Predictions_train,labels=[0,1,2,3,4])
    cm_test  = confusion_matrix(y_test , Predictions_test ,labels=[0,1,2,3,4])
    
    print('   confusion matrix for train data')
    print(cm_train)
    print('   confusion matrix for test data ')
    print(cm_test )
    
    print('****************************************')
    print('****************************************')
    print('****************************************')
    
#%%
    
knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(X_train_reduced, y_train)

Predictions_train = knn_classifier.predict(X_train_reduced)
Predictions_test  = knn_classifier.predict(X_test_reduced )

accuracy_train = f1_score(Predictions_train,y_train, average='micro')
accuracy_test  = f1_score(Predictions_test, y_test , average='micro')

#print('for n_neighbors = '+str(k)+': ')

print('   f1_score_train: '+str(accuracy_train))
print('   f1_score_test : '+str(accuracy_test ))

cm_train = confusion_matrix(y_train, Predictions_train,labels=[0,1,2,3,4])
cm_test  = confusion_matrix(y_test , Predictions_test ,labels=[0,1,2,3,4])

print('   confusion matrix for train data')
print(cm_train)
print('   confusion matrix for test data ')
print(cm_test )

print('****************************************')
print('****************************************')
print('****************************************')



#%% testing with 20newsgroup dataset
from sklearn.datasets import fetch_20newsgroups
categories = ['talk.politics.misc','rec.sport.baseball','sci.electronics']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
#politics 2, sports 0->3, 1->4
documents2 = []
y2 = []
for i in range(len(newsgroups_train.target)):
    documents2.append(newsgroups_train.data[i])
    y2.append(newsgroups_train.target[i])

#### cambiando etiquetas de 20newsgroup a las propias
for i in range(len(y2)):
    if  y2[i] == 0:
        y2[i] = 3
    elif y2[i] == 1:
        y2[i] = 4
    else:
        y2[i] = 2

DB_size = len(documents2)
print('tamaño de la DB: '+str(DB_size))

X = TFIDFvectorizer.transform(documents2)
X = scaler.transform(X.A)
X = pca.transform(X)

knn_prediction = knn_classifier.predict(X)
svm_prediction = svm_classifier.predict(X)


for i in range(len(knn_prediction)):
    print('label: '+str(y2[i])+'; w/KNN: '+str(knn_prediction[i])+'; w/SVM: '+str(svm_prediction[i]))

print('f1_score w/ KNN: '+str(100*f1_score(y2,knn_prediction, average='micro'))+'%')
print('f1_score w/ SVM: '+str(100*f1_score(y2,svm_prediction, average='micro'))+'%')
# %%


# %%

#with open(r'D:\UdeA\Trabajo_de_grado\2018_2\17_10_18\Training_classification_models\knn_model_2', 'wb') as fp:
#    pickle.dump(clf, fp)
#    fp.close()
#
## %%
#with open(r'D:\UdeA\Trabajo_de_grado\2018_2\17_10_18\Training_classification_models\pca_model_2', 'wb') as fp:
#    pickle.dump(pca, fp)
#    fp.close()
#    
#with open(r'D:\UdeA\Trabajo_de_grado\2018_2\17_10_18\Training_classification_models\scaler_2', 'wb') as fp:
#    pickle.dump(scaler, fp)
#    fp.close()
#(1891, 10408)
    
# %%
    
#indexes = [i for i in range(X.shape[0])]
#random.shuffle(indexes)
#
#with open(r'D:\UdeA\Trabajo_de_grado\2018_2\17_10_18\Training_classification_models\indexes', 'wb') as fp:
#    pickle.dump(indexes, fp)
#    fp.close()

