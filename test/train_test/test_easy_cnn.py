import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from keras.datasets import mnist

from src.model.cnn_model import TestConvNet
from torchsummary import summary
from torch.utils.data import DataLoader
from chun_img_browser.img_browser import ImgBrowser
from src.data.load_data.load_data import MRIDataset
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
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        featuresTrain = torch.from_numpy(X_train)
        targetsTrain = torch.from_numpy(Y_train) # data type is long

        featuresTrain = featuresTrain.unsqueeze(1)

        featuresTest = torch.from_numpy(X_test)
        targetsTest = torch.from_numpy(Y_test) # data type is long
        featuresTest = featuresTest.unsqueeze(1)

        train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
        test = torch.utils.data.TensorDataset(featuresTest,targetsTest)
        #test_data_dict = data['Dataset_path_test']
        #test_dataset = MRIDataset(test_data_dict)
        #
        #train_data_dict = data['Dataset_path_train']
        #train_dataset = MRIDataset(train_data_dict)

        dataloader = {'train':DataLoader(train,batch_size=10,shuffle=True),'val':DataLoader(test,batch_size=10,shuffle=True)}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dwt_cnn = TestConvNet().to(device)
        #summary(dwt_cnn,(3,26,22))
        criterion = nn.CrossEntropyLoss()

        #for inputs, label in dataloader['train']:
        #    print(inputs.shape)
        #    print(label)
        #    inputs = inputs.to(device,dtype=torch.float)
        #    labels = label.to(device)
        #    outputs = dwt_cnn(inputs)
        #    loss = criterion(outputs, labels)
        #    print(loss)

            #break
        optimizer_ft = optim.SGD(dwt_cnn.parameters(),lr=0.001,momentum=0.9)

        dwt_train = DwtConvTrain(dwt_cnn,dataloader,criterion,optimizer_ft,epochs=20)

        dwt_train.fit()


