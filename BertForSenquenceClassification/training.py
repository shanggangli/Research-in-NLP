#-*- codeing=utf-8 -*-
#@time: 2020/8/19 12:47
#@Author: Shang-gang Lee

#accuracy
import numpy as np
import torch
from transformers import AdamW,get_linear_schedule_with_warmup
import time
import datetime
def flat_accuracy(preds,labels):
    pred_falt=np.argmax(preds,axis=1).flatten()
    labels_falt=labels.flatten()
    return np.sum(pred_falt==labels_falt)/len(labels_falt)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

'''
Bert recommender:
Batch size: 16, 32
Learning rate (Adam): 5e-5, 3e-5, 2e-5
Number of epochs: 2, 3, 4
'''
#train
def train(model,train_loader,validation_dataloader):
    # optimiter
    device=('cuda' if torch.cuda.is_available() else 'gpu')
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    training_stats=[]
    for i in range(epochs):
        print("==========training=============")
        print('Epoch:{:}/{:}'.format(i + 1, epochs))

        t0=time.time()
        total_train_loss=0
        model.train()
        for step,batch in enumerate(train_loader):
            b_input_ids=batch[0].to(device)
            b_attention_mask=batch[1].to(device)
            b_label=batch[2].to(device)
            model.zero_grad()
            loss,logit=model(input_ids=b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_attention_mask,
                             labels=b_label)
            total_train_loss+=loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        Avg_train_loss=total_train_loss/len(train_loader)
        train_time=format_time(time.time()-t0)
        print('Average training loss:{0:.2f}'.format({Avg_train_loss}))
        print('training time:{:}'.format((train_time)))

        print("============Running Validation===========")
        model.eval()
        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        t1 = time.time()
        for batch in validation_dataloader:
            b_input_ids=batch[0].to(device)
            b_attenion_mark=batch[1].to(device)
            b_label=batch[2].to(device)

            with torch.no_grad():
                loss,logit=model(input_ids=b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_attenion_mark,
                                 labels=b_label)
                total_eval_loss+=loss.item()
                logit=logit.detach().cpu().numpy()
                label_ids=b_label.to('cpu').numpy()
                total_eval_accuracy+=flat_accuracy(logit,label_ids)

        eval_time=format_time(time.time()-t1)
        Avg_eval_accuracy=total_eval_accuracy/len(validation_dataloader)
        Avg_eval_loss=total_eval_loss/len(validation_dataloader)
        print("accuracy:{0:.2f}".format(Avg_eval_accuracy))
        print('validation loss:{0:.2f}'.format(Avg_eval_loss))
        print('validation time:{:}'.format(eval_time))
        training_stats.append(
            {
                'epoch': i+1,
                'Training Loss': Avg_train_loss,
                'Valid. Loss': Avg_eval_loss,
                'Valid. Accur.': Avg_eval_accuracy,
                'Training Time': train_time,
                'Validation Time': eval_time
            }
        )
    print("")
    print("Training complete!")
    return training_stats