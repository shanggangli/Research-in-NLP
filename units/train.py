#-*- codeing=utf-8 -*-
#@time: 2020/8/26 11:32
#@Author: Shang-gang Lee
import torch
import numpy as np

def train(model,data_loader,val_loader,optimizer,loss_function):
    device=torch.device('cuda' if torch.cuda.is_available() else 'gpu')
    model=model.to(device)

    def flat_accuracy(preds, labels):
        pred_falt = np.argmax(preds, axis=1).flatten()
        labels_falt = labels.flatten()
        return np.sum(pred_falt == labels_falt) / len(labels_falt)
    loss_total=0
    epochs=4

    for i in range(epochs):
        print("#########training#############")
        print("#######Epoch:{:}/{:}#########".format(i+1,epochs))
        for step,(input,label) in enumerate(data_loader):
            input=input.to(device)
            label=label.to(device)
            pred=model(input)
            loss=loss_function(pred,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total+=loss

            if step%1000==0:
                print("============Running Validation===========")
                with torch.no_grad():
                    total_val_accuracy=0
                    for val_input,val_label in val_loader:
                        val_input=val_input.to(device)
                        val_label=val_label.to(device)
                        val_pred=model(val_input)
                        total_val_accuracy+=flat_accuracy(val_pred,val_label)
                    Avg_val_accuracy=total_val_accuracy/len(val_loader)
                    print('step:{:}    Average accuracy:{:}'.format(step,Avg_val_accuracy))