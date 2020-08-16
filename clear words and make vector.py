#-*- codeing=utf-8 -*-
#@time: 2020/8/16 13:54
#@Author: Shang-gang Lee

import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer

train_data=pd.read_csv(r'./data/train.csv')

def sentence_to_word(data):
    ps = PorterStemmer()
    letter = re.sub('https?://\S+|www\.\S+', '', data)    #清除网址
    words= re.sub(r'[^A-Za-z]+', ' ', letter).lower().split() # 清除不是字母的字符
    stop = set(stopwords.words('english'))
    clear_text = [ps.stem(w) for w in words if not w in stop] #词性还原+去除stopword
    if len(clear_text)>2:
        return clear_text

train_data['word']=train_data['text'].apply(str).apply(lambda x:sentence_to_word(x))
train_data.dropna(axis=0,how='any',inplace=True)

path = r'.\glove.6B.100d.txt'
with open(path, 'r', errors='ignore') as f:
    line = list(f)
    embedding_dict = {}
    for i in line:
        values = i.split()
        word = values[0]
        embedding_dict[word] = np.asarray(values[1:], 'float32')
    f.close()
    
word_vector=np.zeros((100,),dtype='float32')

def word_to_vector(words_list,num_feature):
    count=0
    word_vector = np.zeros((num_feature,), dtype='float32')
    for i in words_list:
        if i in embedding_dict:
            word_vector=np.add(word_vector,embedding_dict[i])
            count+=1
    word_Avgvector=np.divide(word_vector,count)
    return word_Avgvector

embedding_vector=np.zeros((len(train_data['word']),100),dtype='float32')

count=0
for i in train_data['word']:
    embedding_vector[count]=word_to_vector(i,100)
    count+=1
print(embedding_vector)
