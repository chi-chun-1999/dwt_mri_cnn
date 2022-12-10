from torch.utils.data import Dataset
import torch
import yaml
import nibabel as nib
import os
from glob import glob
from operator import add 
from functools import reduce
import numpy as np
from ..transfer.image_transfer import ImgFourierTransfer
from ..transfer.image_transfer import DiscreteWaveletTransform
from skimage import transform

class Produce2dDataset():
    def __init__(self,axis='z'):

        if axis == 'x' or axis == 'X':
            self._axis = 0
        elif axis == 'y' or axis == 'Y':
            self._axis = 1
        elif axis == 'z' or axis == 'Z':
            self._axis = 2

        self._input_dataset = None
        self._label_dataset = None

    def get_data(self):
        return self._input_dataset,self._label_dataset

    def produce_data(self,pt_input_data,pt_label_data):
        slice_each_pt_input = torch.unbind(pt_input_data)
        slice_each_pt_label = torch.unbind(pt_label_data)


        iter_num = 0

        for each_input,each_label in zip(slice_each_pt_input,slice_each_pt_label):
            for slice_from_axis in torch.unbind(each_input,dim=self._axis):
                if iter_num == 0:
                    slice_from_axis = torch.unsqueeze(slice_from_axis,0)
                    self._input_dataset = torch.unsqueeze(slice_from_axis,0)
                    _,each_label_without_one_hot = each_label.max(dim=0)
                    self._label_dataset = torch.unsqueeze(each_label_without_one_hot,0)
                    iter_num += 1
                else:
                    _,each_label_without_one_hot = each_label.max(dim=0)
                    slice_from_axis = torch.unsqueeze(slice_from_axis,0)
                    unsqueeze_input_data = torch.unsqueeze(slice_from_axis,0)
                    unsqueeze_label_data = torch.unsqueeze(each_label_without_one_hot.max(),0)
                    self._input_dataset = torch.cat((self._input_dataset,unsqueeze_input_data),0)
                    self._label_dataset = torch.cat((self._label_dataset,unsqueeze_label_data),0)


        #print(self._input_dataset.shape)
        #print(self._label_dataset.shape)

                    

class ProduceDWTfrom2d():
    def __init__(self,dwt_wavelet='db32',return_type='LL'):
        self._dwt_wavelet = dwt_wavelet
        self._return_type = return_type
        self._input_dataset = None
        self._label_dataset = None
        self._discrete_wavelet_transform = DiscreteWaveletTransform(self._dwt_wavelet)

    def get_data(self):
        return self._input_dataset,self._label_dataset

    def produce_data(self,pt_input_data,pt_label_data):
        slice_each_pt_input = torch.unbind(pt_input_data)
        slice_each_pt_label = torch.unbind(pt_label_data)


        iter_num = 0

        for each_input,each_label in zip(slice_each_pt_input,slice_each_pt_label):
            tmp_input = torch.squeeze(each_input)
            LL,(LH,HL,HH) = self._discrete_wavelet_transform.process(tmp_input)
            tmp_input_tensor = None

            if self._return_type == 'LL':
                tmp_input_tensor = torch.from_numpy(LL)
            elif self._return_type == 'LH':
                tmp_input_tensor = torch.from_numpy(LH)
            elif self._return_type == 'HL':
                tmp_input_tensor = torch.from_numpy(HL)
            elif self._return_type == 'HH':
                tmp_input_tensor = torch.from_numpy(HH)


            if iter_num == 0:

                tmp_input_tensor = torch.unsqueeze(tmp_input_tensor,0)
                self._input_dataset = torch.unsqueeze(tmp_input_tensor,0)
                self._label_dataset = torch.unsqueeze(each_label,0)
                iter_num += 1
            else:
                tmp_input_tensor = torch.unsqueeze(tmp_input_tensor,0)
                unsqueeze_input_data = torch.unsqueeze(tmp_input_tensor,0)
                unsqueeze_label_data = torch.unsqueeze(each_label.max(),0)
                self._input_dataset = torch.cat((self._input_dataset,unsqueeze_input_data),0)
                self._label_dataset = torch.cat((self._label_dataset,unsqueeze_label_data),0)

                

class ProduceOneSlice2dDataset():
    def __init__(self,slice_num ,axis='z'):

        if axis == 'x' or axis == 'X':
            self._axis = 0
        elif axis == 'y' or axis == 'Y':
            self._axis = 1
        elif axis == 'z' or axis == 'Z':
            self._axis = 2

        self._input_dataset = None
        self._label_dataset = None
        self._slice_num = slice_num

    def get_data(self):
        return self._input_dataset,self._label_dataset

    def produce_data(self,pt_input_data,pt_label_data):
        slice_each_pt_input = torch.unbind(pt_input_data)
        slice_each_pt_label = torch.unbind(pt_label_data)

        if self._axis == 0:
            self._input_dataset=pt_input_data[:,self._slice_num,:,:]
        elif self._axis == 1:
            self._input_dataset=pt_input_data[:,:,self._slice_num,:]
        elif self._axis == 2:
            self._input_dataset=pt_input_data[:,:,:,self._slice_num]

        self._input_dataset = torch.unsqueeze(self._input_dataset,1)

        _,self._label_dataset = pt_label_data.max(dim=1)

        
