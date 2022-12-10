import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch

from src.model.cnn_model import DwtConvNet4Class
from torchsummary import summary
from torch.utils.data import DataLoader
from chun_img_browser.img_browser import ImgBrowser
from src.data.load_data.load_data import MRIDataset
import yaml
import numpy as np

if __name__ == '__main__':
    with open('../parameter.yml', 'r') as parameter_yml: 

        #dwt_transform = DiscreteWaveletTransform(times=3)
        imgb = ImgBrowser()
        data = yaml.load(parameter_yml,Loader=yaml.CLoader)

        test_data_dict = data['Dataset_path_test']
        test_dataset = MRIDataset(test_data_dict)
        
        train_data_dict = data['Dataset_path_train']
        train_dataset = MRIDataset(train_data_dict)

        dataloader = {'train':DataLoader(train_dataset,batch_size=10,shuffle=True),'val':DataLoader(test_dataset,batch_size=10,shuffle=True)}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dwt_cnn = DwtConvNet4Class().to(device)
        summary(dwt_cnn,(3,26,22))

        for inputs, label in dataloader['val']:
            print(inputs.shape)
            print(label)
            break



