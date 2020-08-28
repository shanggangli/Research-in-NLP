#-*- codeing=utf-8 -*-
#@time: 2020/8/26 11:28
#@Author: Shang-gang Lee
from torch.utils.data import TensorDataset,random_split

def dataSet(data,label):
    data=TensorDataset(data,label)
    train_size=int(len(data)*0.8)
    val_size=len(data)-train_size
    train_data,val_data=random_split(data,[train_size,val_size])
    return train_data,val_data