#-*- codeing=utf-8 -*-
#@time: 2020/7/24 15:38
#@Author: Shang-gang Lee

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch.optim as optim
deivce=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

raw_text='The announcements have come on three consecutive nights of revitalised White House coronavirus briefings. ' \
         'In this iteration with the president flying solo, and not flanked by his medical advisers. ' \
         'But they have also been much more disciplined than when the president would spend a couple of hours at the lectern, ' \
         'musing on anything and everything - most memorably on whether disinfectant ' \
         'and sunlight should be injected into the body to treat coronavirus.'
#print(raw_text)
def text_listwords(raw_text):
    latter=re.sub('https?://\S+|www\.\S+', '', raw_text)
    latter=re.sub("[^a-zA-Z]",' ',latter)
    latter=re.sub("   *"," ",latter)
    clear_text=latter.lower().split()
    return clear_text
wordlist=text_listwords(raw_text) #split words

# print(wordlist)
# # print(len(wordlist))

def make_context(text,context_num):
    data=[]
    js = [i for i in range(-context_num, context_num + 1) if i != 0] #skip window
    for i in range(context_num,len(text)-context_num):
        context=[text[k+i] for k in js]
        taget=text[i]
        data.append((context,taget))
    return data

# data=make_context(wordlist,2)
# print(data[:5])

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

#print(make_context_vector(data[0][0],word_to_ix))

class CBOW(nn.Module):
    def __init__(self,vocab_num,embed_dim,context_size):
        super(CBOW, self).__init__()
        self.vocab_num=vocab_num
        self.embed_dim=embed_dim
        self.context_size=context_size
        self.embedding=nn.Embedding(num_embeddings=vocab_num,embedding_dim=embed_dim)
        self.linear1=nn.Linear(2*context_size*embed_dim,128)
        self.linear2=nn.Linear(128,vocab_num)

    def forward(self,input):
        embed=self.embedding(input).view(1,-1)
        output=F.relu(self.linear1(embed))
        output=self.linear2(output)
        log_probs=F.log_softmax(output,dim=1)
        return log_probs

def train(model,data):
    losses=[]
    log_probs=None
    loss_func=nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    word_to_ix = {word: i for i, word in enumerate(wordlist)}
    for epoch in range(200):
        total_loss=0
        context_one_hot=[]
        for context,target in data:
            context_vetor=make_context_vector(context,word_to_ix).to(deivce)
            target=torch.tensor([word_to_ix[target]],dtype=torch.long).to(deivce)

            log_probs=model(context_vetor)
            loss=loss_func(log_probs,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
        print("epoch:",epoch,'|total_loss:',total_loss)
        losses.append(total_loss)
    return log_probs
#train

#hpyer parameters
vocab_size=len(wordlist)
embed_dim=10
context_size=2
if __name__ == '__main__':
    model=CBOW(vocab_size,embed_dim,context_size)
    data=make_context(wordlist,context_size)
    word_vetor=train(model,data)
    print(word_vetor.shape)