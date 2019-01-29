#!/usr/bin/env python
# coding: utf-8

# In[1]:


import db_info_extracton as db
import pickle
import time

all_files = []
all_files.append(db.BBC_Dataset_setup('/home/jparango/Documents/Trabajo_de_grado_2019_1/Dataset/bbc/business'))
all_files.append(db.BBC_Dataset_setup('/home/jparango/Documents/Trabajo_de_grado_2019_1/Dataset/bbc/entertainment'))
all_files.append(db.BBC_Dataset_setup('/home/jparango/Documents/Trabajo_de_grado_2019_1/Dataset/bbc/politics'))
all_files.append(db.BBC_Dataset_setup('/home/jparango/Documents/Trabajo_de_grado_2019_1/Dataset/bbc/tech'))
all_files.append(db.BBC_Dataset_setup('/home/jparango/Documents/Trabajo_de_grado_2019_1/Dataset/bbc/sport'))

DB_size = 0
for i in range(len(all_files)):
    DB_size += len(all_files[i])

# In[4]:

docs = ''
m = len(all_files)
start = time.time()
for i in range(m):
    n = len(all_files[i])
    for j in range(n):
        print('label '+str(i+1)+', text_file '+str(j+1))
        docs += all_files[i][j]
        
# %%
    
db_tokens = db.myTextPreprocessing(docs)
print('****************')
print(db_tokens)
print('****************')
print('Elapsed time: '+str(time.time()-start)+' [sec]')


# In[ ]:

with open('db_file_W_stop_words_29_01_2019', 'wb') as fp:
    pickle.dump(db_tokens, fp)
    fp.close()

