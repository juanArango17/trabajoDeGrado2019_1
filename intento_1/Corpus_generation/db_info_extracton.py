import os
from nltk import sent_tokenize

from nltk.tokenize import ToktokTokenizer
from nltk.corpus import words, stopwords
from nltk.stem import WordNetLemmatizer

toktok = ToktokTokenizer()
wordnet_lemmatizer = WordNetLemmatizer()
string_corpus = sorted(words.words())
stopWords = set(stopwords.words('english'))

def int_concat(integer):
    if(integer>99 and integer <1000): 
        return str(integer)
    elif(integer>9 and integer <100):
        return '0'+str(integer)
    else:
        return '00'+str(integer)

def BBC_Dataset_setup(folder = '/home/jparango/Documents/Trabajo_de_grado_2019_1/Dataset/bbc/business'):

    file_list = os.listdir(folder) # dir is your directory path
    number_files = len(file_list)
    data_list = []
    for i in range(1,number_files+1):
        name = int_concat(i)
        #print(name+'.txt')
        file = name+'.txt'
        data_list.append(open(folder+'/'+file,'r',encoding='latin2').read())
        #print(data_list[0].read())
    return data_list
	
def myTextPreprocessing(text):
    ### Tokenizing
    #print('Tokenizing')
    tokens = [toktok.tokenize(sent) for sent in sent_tokenize(text)]
    flattened_tokens = []
    for sublist in tokens:
        for val in sublist:
            flattened_tokens.append(val)
    #delete marks (@ . , ; :) with_no_marks_and_stopwords
    #print('delete marks/lowercase/deleting stopwords & lemmatize for len(tokens) = '+str(len(flattened_tokens)))
    
    filtered_words = []
    cnt = 0
    for x in flattened_tokens:
        lemm_word = wordnet_lemmatizer.lemmatize(x.lower(),pos='v')
        if x.lower() in string_corpus and x.lower() not in stopWords and lemm_word not in filtered_words:
            filtered_words.append(lemm_word)
        print('filtered words = '+str(len(filtered_words))+' of '+str(len(flattened_tokens))
            +', '+str(100*float(cnt)/len(flattened_tokens))+'%')
        cnt += 1

    return filtered_words

def myTokenizer(text,dbWords):
    tokens = [toktok.tokenize(sent) for sent in sent_tokenize(text)]
    #print('tokened')
    flattened = []
    for sublist in tokens:
        for val in sublist:
            flattened.append(val)
    #print('flattened')
    docs_words = []
    
    for w in flattened:
        if w.lower() in dbWords:
            docs_words.append(w.lower())
    return docs_words
