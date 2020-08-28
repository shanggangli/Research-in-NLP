#-*- codeing=utf-8 -*-
#@time: 2020/8/26 9:49
#@Author: Shang-gang Lee
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class fasttext(nn.Module):
    def __init__(self, vocab_size, embedding_dim, class_nums):
        super().__init__()

        self.embedding=nn.Embedding(vocab_size,embedding_dim)       #[batch_size,words,embedding_dims]
        self.hidden=nn.AdaptiveAvgPool2d((1,embedding_dim))         #[batch_size,embedding_dims]
        self.fc=nn.Linear(embedding_dim,class_nums)                 #[batch_size,class_nums]

    def forward(self,x):
        embed=self.embedding(x)

        avgEmbed=self.hidden(embed)
        avgEmbed=avgEmbed.squeeze(dim=1)

        output=self.fc(avgEmbed)
        output=F.softmax(output,dim=1)

        return output
model=fasttext(vocab_size=2000,embedding_dim=100,class_nums=5)
input=torch.randint(low=0,high=2000,size=[64,50])
pred=model(input)

loss_fucntion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.001)

