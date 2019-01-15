# -*- coding: utf-8 -*-

"""
Created on Mon Oct 29 14:29:57 2018

@author: user
"""


import db_info_extraction as db
import numpy as np
import random

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

#TFIDFvectorizer = TfidfVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1,encoding='latin1')
TFIDFvectorizer = TfidfVectorizer(ngram_range = (1,2),token_pattern=r'\b\w+\b',encoding='latin1')
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

# %%
def remove_zero_tf_idf(Xtr, min_tfidf=0.04):
    Xtr[Xtr < min_tfidf] = 0
    tfidf_means = np.mean(Xtr, axis=0) # find features that are 0 in all documents
    Xtr = np.delete(Xtr, np.where(tfidf_means == 0)[0], axis=1) # delete them from the matrix
    return [Xtr,np.where(tfidf_means == 0)[0]]

# %%
#list(nltk.bigrams(text.split()))
#%%Processing info
indexes = [i for i in range(len(documents))]
random.shuffle(indexes)

doc_rand = documents
y_random = np.copy(y)
for i in range(len(indexes)):
    doc_rand[i] = documents[indexes[i]]
    y_random[i] = y[indexes[i]]
print('fit TFIDFvectorizer')
TFIDFvectorizer.fit(documents)
vectors = TFIDFvectorizer.transform(doc_rand)
vectors = vectors.A
print('# features: '+str(vectors.shape[1]))

print('store test data')    
X, X_test, y, y_test = train_test_split(vectors, y_random, test_size=0.1, random_state=0)

#%% KNN Training Algorithm

print('*********')
print('Running KNN Algorithm')
print('*********')

SCORES_TEST = []
SCORES_TRAIN = []
ALL_KNN_MODELS = []
ALL_PCA_MODELS = []
ALL_SCALERS    = []

for neigh in range(4,7):
    kf = KFold(n_splits=10)

    cnt = 0
    best_score = 0
    mean_score_train = 0
    mean_score_test  = 0
    
    print('using K = '+str(neigh))
    for train_index, test_index in kf.split(X): 
        print('     validation #'+str(cnt+1))
        X1,X2,y1,y2 = X[train_index],X[test_index],y[train_index],y[test_index]
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X1)
        X_test_norm  = scaler.transform(X2)
        pca    = PCA(n_components = 0.9, svd_solver = 'full')
        X_train_reduced = pca.fit_transform(X_train_norm)
        X_test_reduced  = pca.transform(X_test_norm)
        
        knn_classifier = KNeighborsClassifier(n_neighbors = neigh)
        knn_classifier.fit(X_train_reduced, y1)
    
        Predictions_train = knn_classifier.predict(X_train_reduced)
        Predictions_test  = knn_classifier.predict(X_test_reduced )
    
        accuracy_train = f1_score(Predictions_train,y1, average='macro')
        accuracy_test  = f1_score(Predictions_test, y2, average='macro')
    
        print('        f1_score_train: '+str(accuracy_train))
        print('        f1_score_test : '+str(accuracy_test ))
        
        mean_score_train += accuracy_train
        mean_score_test  += accuracy_test
        
        if(accuracy_test>best_score):
            print('           ¡¡¡New Model!!!')
            SCALER = scaler
            PCA_Model = pca
            KNN_MODEL = knn_classifier
            best_score = accuracy_test
        cnt += 1
    
    mean_score_train /= cnt
    mean_score_test  /= cnt
    
    print('   mean_f1_score_for_train w/ K = '+str(neigh)+': '+str(mean_score_train))
    print('   mean_f1_score_for_test w/ K = '+str(neigh)+': '+str(mean_score_test ))
    
    SCORES_TRAIN.append(mean_score_train)
    SCORES_TEST.append(mean_score_test)
    ALL_KNN_MODELS.append(KNN_MODEL)
    ALL_PCA_MODELS.append(PCA_Model)
    ALL_SCALERS.append(SCALER)
    
plt.plot(list(range(1,10)),[1.0-x for x in SCORES_TRAIN],'b',label='Train Error')
plt.hold(True)
plt.plot(list(range(1,10)),[1.0-x for x in SCORES_TEST],'r',label='Test Error')
plt.legend()
plt.xlabel('# Neighbors')
plt.ylabel('Error Function')
plt.grid(True)
plt.show()
    
    
#%%
import pandas as pd

d = {'1. K': list(range(1,10)), '2. P_train': SCORES_TRAIN, '3. P_test': SCORES_TEST}
df = pd.DataFrame(data=d)
print(df)

# %% 
best = 0

SCALER = ALL_SCALERS[best]
PCA_Model = ALL_PCA_MODELS[best]
KNN_MODEL = ALL_KNN_MODELS[best]

    
# %%
X_ = SCALER.transform(X)
X_train = PCA_Model.transform(X_)
Predictions_train =KNN_MODEL.predict(X_train)

X_ = SCALER.transform(X_test)
X_t = PCA_Model.transform(X_)
Predictions_test = KNN_MODEL.predict(X_t)

cm_train = confusion_matrix(y, Predictions_train,labels=[0,1,2,3,4])
cm_test = confusion_matrix(y_test, Predictions_test,labels=[0,1,2,3,4])

print('mean_f1_score_for_train:'+str(f1_score(y,Predictions_train, average='macro')))
print('mean_f1_score_for_test :'+str(f1_score(y_test,Predictions_test, average='macro')))

print('confusion matrix for train data ')
print(cm_train )

print('confusion matrix for test data ')
print(cm_test )

#%%
print('*********')
print('Testing with 20newsgroup dataset')
print('*********')

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
        y2[i] = 4
    elif y2[i] == 1:
        y2[i] = 3
    else:
        y2[i] = 2

DB_size = len(documents2)
print('tamaño de la DB: '+str(DB_size))

X = TFIDFvectorizer.transform(documents2)
X = SCALER.transform(X.A)
X = PCA_Model.transform(X)


knn_prediction = KNN_MODEL.predict(X)

"""
for i in range(len(knn_prediction)):
    print('label: '+stropen(folder+'/'+file,'r')(y2[i])+'; w/KNN: '+str(knn_prediction[i]))
"""

print('f1_score w/ KNN: '+str(100*f1_score(y2,knn_prediction, average='micro'))+'%')

# %%

with open('/home/jparango/Documents/Trabajo_de_grado_2019_1/intento_2___con_bigramas/objects_results/knn_model_2', 'wb') as fp:
    pickle.dump(KNN_MODEL, fp)
    fp.close()

with open('/home/jparango/Documents/Trabajo_de_grado_2019_1/intento_2___con_bigramas/objects_results/pca_model_2', 'wb') as fp:
    pickle.dump(PCA_Model, fp)
    fp.close()
    
with open('/home/jparango/Documents/Trabajo_de_grado_2019_1/intento_2___con_bigramas/objects_results/scaler_2', 'wb') as fp:
    pickle.dump(SCALER, fp)
    fp.close()

with open('/home/jparango/Documents/Trabajo_de_grado_2019_1/intento_2___con_bigramas/objects_results/tfidfvectorizer', 'wb') as fp:
    pickle.dump(TFIDFvectorizer, fp)
    fp.close()
#(1891, 10408)
    
# %%
    
#indexes = [i for i in range(X.shape[0])]
#random.shuffle(indexes)
#
#with open(r'D:\UdeA\Trabajo_de_grado\2018_2\17_10_18\Training_classification_models\indexes', 'wb') as fp:
#    pickle.dump(indexes, fp)
#    fp.close()




# %%
"""
[X_removed, col_removed] = remove_zero_tf_idf(X)
print('TFIDFvectorizer.size: '+str(X_removed.shape))
print('TFIDFvectorizer.col_removed: '+str(col_removed))
"""




