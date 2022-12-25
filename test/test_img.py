import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision import datasets,transforms
import torch
from torch.utils.data import DataLoader
from chun_img_browser.img_browser import ImgBrowser
from src.data.transform.image_transform import DiscreteWaveletTransform
from src.data.load_data.load_data import get_jpg_path
from src.data.load_data.load_data import MRIDatasetResize
import yaml
import numpy as np


if __name__ == "__main__":
    with open('../parameter.yml', 'r') as parameter_yml: 

        #dwt_transform = DiscreteWaveletTransform(times=3)
        imgb = ImgBrowser()
        data = yaml.load(parameter_yml,Loader=yaml.CLoader)

        test_data_dict = data['Dataset_path_test']
        test_dataset = MRIDatasetResize(test_data_dict,dwt_times=3)
        
        train_data_dict = data['Dataset_path_train']
        train_dataset = MRIDatasetResize(train_data_dict,dwt_times=3)

        train_dataloader = DataLoader(train_dataset,batch_size=10,shuffle=True)
        test_dataloader = DataLoader(test_dataset,batch_size=10,shuffle=True)

        for inputs, label in train_dataloader:
            print(inputs.shape)
            print(label)
            break

        #imgb.imshow(np.array(test_dataset[0]))
        



