#-*- codeing=utf-8 -*-
#@time: 2020/8/19 12:52
#@Author: Shang-gang Lee

import build_model
import ProcessingData
import training
import pandas as pd
import torch
if __name__ == '__main__':
    # data
    train_data = pd.read_csv(r'.\data\raw\in_domain_train.tsv',
                             delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    dev_data = pd.read_csv(r'.\data\raw\in_domain_dev.tsv',
                           delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    # print('train_data.shape:',train_data.shape)
    # print('train_data.isnull.size:',train_data.isnull().sum())
    # print('dev_data.shape:',dev_data.shape)
    # print('dev_data.isnull.size:',dev_data.isnull().sum())
    sentences = train_data.sentence.values
    labels = train_data.label.values

    #getting inputdata
    input_ids, attention_masks =ProcessingData.get_inputdata(sentences)
    labels = torch.tensor(labels)

    #get loader
    train_loader, validation_dataloader=ProcessingData.get_loader(input_ids, attention_masks,labels)

    #build model
    model=build_model.build_model()

    #training
    train_result=training.train(model,train_loader,validation_dataloader)

    print(train_result)