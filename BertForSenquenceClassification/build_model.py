#-*- codeing=utf-8 -*-
#@time: 2020/8/19 12:47
#@Author: Shang-gang Lee

import torch
from transformers import BertForSequenceClassification
def build_model(num_labels=2):
    #build model
    model=BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                        num_labels=num_labels,
                                                        output_attentions=False,
                                                        output_hidden_states=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model

