import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch

from src.model.cnn_model import DwtConvNet2Class
from src.model.cnn_model import DwtConvNet2Class5Block
from torchsummary import summary
from torch.utils.data import DataLoader
from chun_img_browser.img_browser import ImgBrowser
from src.data.load_data.load_data import MRIDatasetResize
from src.train.train import DwtConvTrain
from torch.utils.data import TensorDataset
import yaml
import numpy as np
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    with open('../parameter.yml', 'r') as parameter_yml: 

        #dwt_transform = DiscreteWaveletTransform(times=3)
        imgb = ImgBrowser()
        data = yaml.load(parameter_yml,Loader=yaml.CLoader)

        dwt_times = 3

        data_dict = data['ArrayData']['2class_data']['haar_3']

        train_inputs = torch.load(data_dict['resize_train_inputs'])
        train_labels = torch.load(data_dict['resize_train_labels'])
        val_inputs = torch.load(data_dict['resize_val_inputs'])
        val_labels = torch.load(data_dict['resize_val_labels'])


        train_dataset = TensorDataset(train_inputs,train_labels)
        
        val_dataset =TensorDataset(val_inputs,val_labels) 

        dataloader = {'train':DataLoader(train_dataset,batch_size=10,shuffle=True),'val':DataLoader(val_dataset,batch_size=10,shuffle=True)}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dwt_cnn = DwtConvNet2Class5Block().to(device)
        summary(dwt_cnn,(3,64,64))
        #summary(dwt_cnn,(3,104,88))
        criterion = nn.CrossEntropyLoss()

        #for inputs, label in dataloader['train']:
        #    print(inputs.shape)
        #    print(label)
        #    #inputs = inputs.to(device,dtype=torch.float)
        #    #labels = label.to(device)
        #    #outputs = dwt_cnn(inputs)
        #    #loss = criterion(outputs, labels)
        #    #print(loss)

        #    break
        optimizer_ft = optim.SGD(dwt_cnn.parameters(),lr=0.005,momentum=0.9)

        dwt_train = DwtConvTrain(dwt_cnn,dataloader,criterion,optimizer_ft,epochs=500)

        dwt_train.fit()


