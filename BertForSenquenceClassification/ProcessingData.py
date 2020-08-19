#-*- codeing=utf-8 -*-
#@time: 2020/8/19 12:48
#@Author: Shang-gang Lee
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset,random_split
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler

def get_inputdata(sentences):
    input_ids=[]
    attention_masks=[]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for sent in sentences:
        encode_dict=tokenizer.encode_plus(sent,add_special_tokens=True,
                              max_length=64,
                              pad_to_max_length=True,
                              return_attention_mask=True,
                                truncation=True,
                              return_tensors='pt')
        input_ids.append(encode_dict['input_ids'])
        attention_masks.append(encode_dict['attention_mask'])
    input_id=torch.cat(input_ids,dim=0)
    attention_masks=torch.cat(attention_masks,dim=0)
    return input_id,attention_masks

def get_loader(input_ids, attention_masks, labels,batch_size=32):
    # TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, lengths=[train_size, val_size])

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset))
    return train_loader,validation_dataloader