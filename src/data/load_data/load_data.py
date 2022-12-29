from torch.utils.data import Dataset
import torch
import yaml
import nibabel as nib
import os
from glob import glob
from operator import add 
from functools import reduce
import numpy as np
from ..transform.image_transform import DiscreteWaveletTransform
from skimage import transform
import matplotlib.image as mbimg
import cv2 as cv


#def get_nii_path(dataset_path_dict):
#
#    img_path_dict = {}
#
#    for key,path in dataset_path_dict.items():
#        if key!='image_root':
#            each_sub_dir = glob(path+'Sub*')
#            #print(each_sub_dir)
#            each_nii_files = [glob(i+'/*.nii') for i in each_sub_dir]
#            #print(path,len(each_nii_files))
#            each_nii_files =reduce(add,each_nii_files)
#            img_path_dict[key]=each_nii_files
#    return img_path_dict

def get_jpg_path(dataset_path_dict):
    img_path_dict = {}
    for key,path in dataset_path_dict.items():
        if key!='image_root':
            each_sub_dir = glob(path)
            #print(each_sub_dir)
            each_jpg_files = [glob(i+'/*.jpg') for i in each_sub_dir]
            #print(path,len(each_nii_files))
            each_jpg_files =reduce(add,each_jpg_files)
            img_path_dict[key]=each_jpg_files
    return img_path_dict




class MRIDataset(Dataset):
    def __init__(self, dataset_path_dict,dwt_times = 1):
        # load all nii handle in a list
        self._img_transfer = DiscreteWaveletTransform(times=dwt_times)
        self._initial_dataset(dataset_path_dict)

        #print(self._label)

    def _initial_dataset(self,dataset_path_dict):
        img_path_dict = get_jpg_path(dataset_path_dict)

        label_num = 0
        
        img_path = []
        label = []

        for i in img_path_dict.keys():

            img_path.append(img_path_dict[i])

            label.append([label_num for k in range(len(img_path_dict[i]))])
            label_num = label_num+1
        

        self._img_path = reduce(add,img_path)
        self._label = reduce(add,label)


    def __len__(self):
        return len(self._img_path)

    def __getitem__(self, idx):
        mri_jpg_image = cv.imread(self._img_path[idx])
        transform_img = self._img_transfer.process(mri_jpg_image)
        data = torch.from_numpy(transform_img)

        target = self._label[idx]
        return data, target
        
class MRIDatasetResize(Dataset):
    def __init__(self, dataset_path_dict,dwt_times = 1):
        # load all nii handle in a list
        self._img_transfer = DiscreteWaveletTransform(times=dwt_times)
        self._initial_dataset(dataset_path_dict)

        #print(self._label)

    def _initial_dataset(self,dataset_path_dict):
        img_path_dict = get_jpg_path(dataset_path_dict)

        label_num = 0
        
        img_path = []
        label = []

        for i in img_path_dict.keys():

            img_path.append(img_path_dict[i])

            label.append([label_num for k in range(len(img_path_dict[i]))])
            label_num = label_num+1
        

        self._img_path = reduce(add,img_path)
        self._label = reduce(add,label)


    def __len__(self):
        return len(self._img_path)

    def __getitem__(self, idx):
        mri_jpg_image = cv.imread(self._img_path[idx])
        mri_jpg_image = cv.resize(mri_jpg_image, (512, 512), interpolation=cv.INTER_AREA)
        transform_img = self._img_transfer.process(mri_jpg_image)
        data = torch.from_numpy(transform_img)

        target = self._label[idx]
        return data, target
class MRIDatasetResizeMedian(Dataset):
    def __init__(self, dataset_path_dict,dwt_times = 1):
        # load all nii handle in a list
        self._img_transfer = DiscreteWaveletTransform(times=dwt_times)
        self._initial_dataset(dataset_path_dict)

        #print(self._label)

    def _initial_dataset(self,dataset_path_dict):
        img_path_dict = get_jpg_path(dataset_path_dict)

        label_num = 0
        
        img_path = []
        label = []

        for i in img_path_dict.keys():

            img_path.append(img_path_dict[i])

            label.append([label_num for k in range(len(img_path_dict[i]))])
            label_num = label_num+1
        

        self._img_path = reduce(add,img_path)
        self._label = reduce(add,label)


    def __len__(self):
        return len(self._img_path)

    def __getitem__(self, idx):
        mri_jpg_image = cv.imread(self._img_path[idx])
        mri_jpg_image = cv.resize(mri_jpg_image, (512, 512), interpolation=cv.INTER_AREA)
        mri_jpg_image = cv.medianBlur(mri_jpg_image,3)
        transform_img = self._img_transfer.process(mri_jpg_image)
        data = torch.from_numpy(transform_img)

        target = self._label[idx]
        return data, target

class MRIDatasetResizeMedianCombine(Dataset):
    def __init__(self, dataset_path_dict,dwt_times = 1):
        # load all nii handle in a list
        self._img_transfer = DiscreteWaveletTransform(times=dwt_times)
        self._initial_dataset(dataset_path_dict)

        #print(self._label)

    def _initial_dataset(self,dataset_path_dict):
        img_path_dict = get_jpg_path(dataset_path_dict)

        label_num = 0
        
        img_path = []
        label = []

        for i in img_path_dict.keys():

            img_path.append(img_path_dict[i])

            if i=='NonDemented':
                label_num = 1
            label.append([label_num for k in range(len(img_path_dict[i]))])
        

        self._img_path = reduce(add,img_path)
        self._label = reduce(add,label)


    def __len__(self):
        return len(self._img_path)

    def __getitem__(self, idx):
        mri_jpg_image = cv.imread(self._img_path[idx])
        mri_jpg_image = cv.resize(mri_jpg_image, (512, 512), interpolation=cv.INTER_AREA)
        mri_jpg_image = cv.medianBlur(mri_jpg_image,3)
        transform_img = self._img_transfer.process(mri_jpg_image)
        data = torch.from_numpy(transform_img)

        target = self._label[idx]
        return data, target
        return data, target
