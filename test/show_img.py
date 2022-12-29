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
import matplotlib.pyplot as plt
import yaml
import numpy as np
import cv2 as cv
import pywt
import numpy as np

#from chun_img_browser.img_browser import ImgBrowser


if __name__ == "__main__":
    with open('../parameter.yml', 'r') as parameter_yml: 
        imgb = ImgBrowser()

        image_path = "/mnt/chi-chun/data_disk/Dataset/kaggle_dataset/alzheimer_dataset/test/VeryMildDemented/26 (44).jpg"
        image = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
        image = image
        plt.subplot(141)
        plt.imshow(image)
        #cv.imshow('ori image',image)
        print(type(image[0,0]))
        image = cv.resize(image, (512, 512), interpolation=cv.INTER_AREA)
        plt.subplot(142)
        plt.imshow(image)
        image = cv.medianBlur(image,3)
        plt.subplot(143)
        plt.imshow(image)
        LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
        plt.subplot(144)
        plt.imshow(LL)
        plt.show()
        #cv.imshow('image',image)
        #cv.imshow('LL',LL)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        #imgb.imshow(LL)
        
        
