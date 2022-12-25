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
        

class MRIDataset1D(MRIDataset):
    def __init__(self,dataset_path_dict,image_transfer_return="ABSOLUTE"):
        super().__init__(dataset_path_dict)
        self._img_transfer = ImgFourierTransfer(dimesion=3,return_type="ABSOLUTE")

    def __getitem__(self, idx):
        nii_image = nib.load(self._img_path[idx])
        data = torch.from_numpy(np.asarray(nii_image.dataobj))

        if len(data.shape)==4:
            data = data[:,:,:,70]

        #data = self._img_transfer.process(transform.resize(data,(61,73,61)))
        data = self._img_transfer.process(transform.resize(data,(61,73,61)))
        data = torch.reshape(data,(-1,))


        target = self._label[idx]
        # find how to retrieve the target
        return data, target

class SliceNormalizeDataset():
    def __init__(self,tensor_input,tensor_label):

        self._input =  tensor_input
        self._label =  tensor_label
    def __len__(self):
        return len(self._label)

    def __getitem__(self, idx):

        #data = self._img_transfer.process(transform.resize(data,(61,73,61)))
        mean, std = torch.mean(self._input[idx].double()),torch.std(self._input[idx].double())
        data = (self._input[idx]-mean)/std


        target = self._label[idx]
        # find how to retrieve the target
        return data, target

