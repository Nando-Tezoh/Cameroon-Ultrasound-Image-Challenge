import glob
import shutil
from collections import Counter
import copy
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn import metrics




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### Functions to assigns each image to its class (splitting function)
### The function also allow to split the dataset to train and test.
### Before you should create two folders, folder '0' and folder '1'
### images_path  = '/content/drive/MyDrive/Cameroon_Zindi_Competition/images/CAMAIRAI BOX READY'
### Here train/1, train/0 and test are located inside the 'drive/MyDrive/Cameroon_Zindi_Competition/'


def create_label(images_path,df,train=True):

    for img in glob.glob(images_path+'/*.jpg'):
        im = img.split('/')[-1].split('.')[0]
        if im in Counter(df.img_IDs.to_list()):
            if train:
                label = df[df.img_IDs == im].target.values[0]
                if label==1:
                    shutil.copy2(images_path+im+'.jpg','train/1')
                else:
                    shutil.copy2(images_path+im+'.jpg','train/0')
            else:
                shutil.copy2(images_path+im+'.jpg','./test')


                   
                    
def train_model(model,criterion,optimizer,scheduler,dataloaders,dataset_sizes,num_epochs=20):
    since = time.time()
    auc_list = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:

                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            auc = 0
            bs = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs1 = F.softmax(outputs,dim=1)
                    max_prob, preds = torch.max(outputs1, dim=1)

                    prods, preds = torch.max(outputs, 1)
                    #print('prodd', max_prob.cpu().detach().numpy(),labels.data.cpu().detach().numpy())
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                

                fpr, tpr, _= metrics.roc_curve(labels.data.cpu().detach().numpy(),                  
                                                 max_prob.cpu().detach().numpy(), pos_label=None)
                #print('vvv',Counter(labels.data.cpu().detach().numpy()))
                #print(labels.data.cpu().detach().numpy())
                #print(metrics.auc(fpr, tpr))
                auc += metrics.auc(fpr, tpr)
                bs+=1
                #print('auc', auc)

                # statistics
                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)
                # auc_list.append(auc)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_auc = auc/bs

            print('{} Loss: {:.4f} Acc: {:.4f} Auc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_auc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'model.best')


        #print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    model.load_state_dict(torch.load('model.best'))
    return model#, auc_list

### focal loss
##https://github.com/ashawkey/FocalLoss.pytorch
##https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
class FocalLoss(torch.nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss  
                    

                    
                
                
                


    