#-*- codeing=utf-8 -*-
#@time: 2020/8/26 11:32
#@Author: Shang-gang Lee
from Dataset import dataSet
from Processing import sentence_to_word
from train import train
from FastTextModel import fasttext
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
if __name__ == '__main__':
    #getting pretrain_word embedding vectors
    path = r'C:\Users\朝花夕拾\Desktop\audio\glove.6B.100d.txt'
    with open(path, 'r', errors='ignore') as f:
        line=list(f)
        word_vector=[]
        iword=[i.split()[0] for i in line]
        for j in line:
            word_vector.append(j.split()[1:])
    f.close()
    #processing data
    data=pd.read_csv(r'C:\untitled\kaggle\tweet-sentiment-extraction\tweet-sentiment\train.csv')
    dic={'neutral':1,'negative':0,'positive':2}
    data['sentiment']=data['sentiment'].map(dic)
    data.dropna(axis=0,how='any',inplace=True)
    data['word']=data['text'].apply(str).apply(lambda x:sentence_to_word(x))
    data.dropna(axis=0,how='any',inplace=True)

    word=data['word']
    word_index=[iword.index(j) for i in word for j in i if j in iword]
    # data
    word_index=torch.from_numpy(np.array(word_index))
    label=torch.from_numpy(np.array(data['sentiment']))
    train_data,val_data=dataSet(word_index,label)
    train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
    val_loader=DataLoader(val_data,batch_size=64,shuffle=True)

    #CNNtext model
    model=fasttext(vocab_size=40000,embedding_dim=100,class_nums=3)

    #updata word embedding vectors
    model.embedding.weight.copy_(torch.from_numpy(np.array(word_vector)))

    #loss and otpmitizer
    loss_function=nn.CrossEntropyLoss()
    optimizer=optim.AdamW(model.parameters(),lr=0.001)

    #train
    train(model,train_loader,val_loader,optimizer,loss_function)