#-*- codeing=utf-8 -*-
#@time: 2020/9/13 23:41
#@Author: Shang-gang Lee

from torchtext import data,datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import spacy
from torch.nn import init
import torchtext
import jieba
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch.nn as nn
import re
import torch.nn.functional as F
import numpy as np
from spacy.tokenizer import Tokenizer
import nltk

DEVICE=torch.device('gpu' if torch.cuda.is_available() else 'cpu')

# train_data=pd.read_csv(r'C:\untitled\kaggle\tweet-sentiment-extraction\tweet-sentiment\train.csv')
# test_data=pd.read_csv(r'C:\untitled\kaggle\tweet-sentiment-extraction\tweet-sentiment\test.csv')
#
# # dic={'neutral':1,'negative':0,'positive':2}
# # train_data['sentiment']=train_data['sentiment'].map(dic)
# train_data.dropna(axis=0,how='any',inplace=True)
#
# test=test_data[['text']]
# test.dropna(axis=0,how='any',inplace=True)
#
# train,val=train_test_split(train_data,test_size=0.2)
#
# train=train[['text','sentiment']]
# val=val[['text','sentiment']]
#
# def sentence_to_word(data):
# 	ps = PorterStemmer()
# 	letter = re.sub('https?://\S+|www\.\S+', '', data)
# 	words = re.sub(r'[^A-Za-z]+', ' ', letter).lower() # 清除不是字母的字符
# 	# stop = set(stopwords.words('english'))
# 	# clear_text = [ps.stem(w) for w in words.split() if not w in stop]
# 	# if len(clear_text) > 2:
# 	#     return clear_text
# 	return words
# #
# train['text']=train['text'].apply(str).apply(lambda x:sentence_to_word(x))
# val['text']=val['text'].apply(str).apply(lambda x:sentence_to_word(x))
# test['text']=test['text'].apply(str).apply(lambda x:sentence_to_word(x))
# train.to_csv('train.csv',index=False)
# test.to_csv('test.csv',index=False)
# val.to_csv('val.csv',index=False)

stopword = set(stopwords.words('english'))
def tokenzier(sentence):
    return[token for token in nltk.word_tokenize(sentence) if token not in stopword]

def Field():
    LABEL=data.Field(sequential=False,use_vocab=True)
    TEXT=data.Field(sequential=True,tokenize=tokenzier,lower=True)
    return LABEL,TEXT

#splits data
def splits_data(TEXT,LABEL):
    train,val=data.TabularDataset.splits(path='.',
                                        format='csv',
                                         train='train.csv',
                                         validation='val.csv',
                                         skip_header=True,
                                         fields=[('text',TEXT),('sentiment',LABEL)])
    test,_=data.TabularDataset.splits(path='./',
                                         test='test.csv',
                                        train='train.csv',
                                        format='csv',
                                         skip_header=True,
                                         fields=[('text',TEXT)])
    return train,val,test

# bulid vocab
def build_vocab(train,val,test,TEXT):
    pretrain_path=r'C:\Users\朝花夕拾\Desktop\audio'
    pretrain_name='glove.6B.100d.txt'
    vectors=torchtext.vocab.Vectors(name=pretrain_name,cache=pretrain_path)
    LABEL.build_vocab(train,val)
    TEXT.build_vocab(train,val,test,vectors=vectors)
    TEXT.vocab.vectors.unk_init=init.xavier_uniform


# dataiter
def dataiter(train_data,val_data,test_data):
    train_iter,val_iter=torchtext.data.BucketIterator.splits((train_data,val_data),
                                                             batch_sizes=(64,64),
                                                             sort_key=lambda x:len(x.text),
                                                             sort_within_batch=False,
                                                             repeat=False,
                                                             device=DEVICE)
    test_iter=torchtext.data.Iterator(test_data,batch_size=64,sort=False,repeat=False,device=DEVICE)
    return train_iter,val_iter,test_iter

# model
class textCNN(nn.Module):
    #hyper parameters
    def __init__(self,vocab_size,
                 embedding_size=100,
                 dropout=0.1,
                 filter_size=(3,4,5),
                 class_num=3,
                 oc=100,    #out_channels
                 ic=1):     #in_channels
        super(textCNN, self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_size)
        self.convs=nn.ModuleList([nn.Conv2d(ic,oc,kernel_size=(k,embedding_size))
                                 for k in filter_size])
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(len(filter_size)*3,class_num)

    def forward(self,x):

        x=self.embedding(x)  #[batch_size,word_size,word_embedding_size]

        x=x.unsqueeze(1)    #[batch_size,in_channels,word_size,word_embedding_size]

        x=[F.relu(conv(x)).squeeze(3) for conv in self.convs]   #[(batch_size,out_channels,word_size),...]*len( filter_size)

        x=[F.max_pool1d(i,i.size(2)).squeeze(2) for i in x]     #[(batch_size,out_channels,),...]*len( filter_size)

        x=torch.cat(x,dim=1)        #[batch_size,len( filter_size)*out_channels]

        x=self.dropout(x)

        logit=self.fc(x)
        output=torch.softmax(logit,dim=1)   #[batch_size,num_classes]
        return output

# train
def training(model,data_loader,val_loader,optimizer,loss_function,device):
    model=model.to(device)
    def flat_accuracy(preds, labels):
        pred_falt = np.argmax(preds, axis=1).flatten()
        labels_falt = labels.flatten()
        return np.sum(pred_falt == labels_falt) / len(labels_falt)
    loss_total=0
    epochs=4
    step=0
    for i in range(epochs):
        print("#########training#############")
        print("#######Epoch:{:}/{:}#########".format(i+1,epochs))
        for batch in data_loader:
            input=batch.text
            print(input)
            label=batch.sentiment
            print(label)
            pred=model(input)
            loss=loss_function(pred,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total+=loss
            step+=1
            if step%1000==0:
                print("============Running Validation===========")
                with torch.no_grad():
                    total_val_accuracy=0
                    for val in val_loader:
                        val_input=val.text
                        val_label=val.sentiment
                        val_pred=model(val_input)
                        total_val_accuracy+=flat_accuracy(val_pred,val_label)
                    Avg_val_accuracy=total_val_accuracy/len(val_loader)
                    print('step:{:}    Average accuracy:{:}'.format(step,Avg_val_accuracy))

if __name__ == '__main__':
    TEXT,LABEL=Field()
    train,val,test=splits_data(TEXT,LABEL)
    build_vocab(train,val,test,TEXT)
    train_loader,val_loader,test_loader=dataiter(train,val,test)
    print(len(TEXT.vocab))
    print(TEXT.vocab.vectors.shape)
    print(TEXT.vocab.stoi['clothe'])
    #
    # for i in train_loader:
    #     print(i)


    textcnn=textCNN(len(TEXT.vocab))
    textcnn.embedding.weight.data.copy_(TEXT.vocab.vectors)
    optimizer=torch.optim.Adam(textcnn.parameters(),lr=0.01)
    loss=nn.CrossEntropyLoss()
    training(textcnn,train_loader,val_loader,optimizer,loss,DEVICE)