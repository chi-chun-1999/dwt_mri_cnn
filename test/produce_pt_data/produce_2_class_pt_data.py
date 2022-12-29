import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch

from src.model.cnn_model import DwtConvNet2Class
from torchsummary import summary
from torch.utils.data import DataLoader
from chun_img_browser.img_browser import ImgBrowser
from src.data.load_data.load_data import MRIDatasetResize
from src.data.load_data.load_data import MRIDataset
from torch.utils.data import TensorDataset
from src.train.train import DwtConvTrain
import yaml
import numpy as np
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    with open('../parameter.yml', 'r') as parameter_yml: 

        #dwt_transform = DiscreteWaveletTransform(times=3)
        imgb = ImgBrowser()
        data = yaml.load(parameter_yml,Loader=yaml.CLoader)

        dwt_times = 2

        test_data_dict = data['2Dataset_path_test']
        test_dataset = MRIDataset(test_data_dict,dwt_times=dwt_times)
        
        train_data_dict = data['2Dataset_path_train']
        train_dataset = MRIDataset(train_data_dict,dwt_times=dwt_times)

        #array_data_2class_data = data['ArrayData']['2class_data']
        #print(array_data_2class_data['train_inputs'])
        #print(array_data_2class_data['train_labels'])

        tmp_dataloader = {'train':DataLoader(train_dataset,batch_size=train_dataset.__len__(),shuffle=True),'val':DataLoader(test_dataset,batch_size=test_dataset.__len__(),shuffle=True)}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dwt_cnn = DwtConvNet2Class().to(device)
        summary(dwt_cnn,(3,64,64))
        #summary(dwt_cnn,(3,104,88))
        criterion = nn.CrossEntropyLoss()

        train_dataset_inputs=None
        train_dataset_labels=None

        val_dataset_inputs=None
        val_dataset_labels=None



        for inputs, label in tmp_dataloader['train']:

            train_dataset_inputs = inputs
            train_dataset_labels = label
            print(train_dataset_inputs.shape)
            print(train_dataset_labels)
            #inputs = inputs.to(device,dtype=torch.float)
            #labels = label.to(device)
            #outputs = dwt_cnn(inputs)
            #loss = criterion(outputs, labels)
            #print(loss)
            #torch.save(inputs,array_data_2class_data['train_inputs'])
            #torch.save(label,array_data_2class_data['train_labels'])
            print('-----------------------------------------------')

        for inputs, label in tmp_dataloader['val']:
            val_dataset_inputs = inputs
            val_dataset_labels = label
            print(val_dataset_inputs.shape)
            print(val_dataset_labels)
            #inputs = inputs.to(device,dtype=torch.float)
            #labels = label.to(device)
            #outputs = dwt_cnn(inputs)
            #loss = criterion(outputs, labels)
            #print(loss)
            #torch.save(inputs,array_data_2class_data['val_inputs'])
            #torch.save(label,array_data_2class_data['val_labels'])


        train_dataset = TensorDataset(train_dataset_inputs,train_dataset_labels)
        val_dataset = TensorDataset(val_dataset_inputs,val_dataset_labels)
        dataloader = {'train':DataLoader(train_dataset,batch_size=10,shuffle=True),'val':DataLoader(val_dataset,batch_size=10,shuffle=True)}

        for inputs, labels in dataloader['val']:
            print(inputs.shape)
            print(labels)
