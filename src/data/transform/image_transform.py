import abc 
import torch.fft as fft
import torch
import pywt
import numpy as np


class TransformMethod(abc.ABC):
    # abstract Class
    def __init__(self):
        self._transfer_data = None

    @abc.abstractmethod
    def process(self):
        return NotImplemented

class ImgFourierTransfer(TransformMethod):

    def __init__(self, dimesion=2,return_type=None):
        """TODO: Docstring for __init__.
        
        Args: 
            - dimesion: the dimesion to use the fft 
            - return_type: the type to return only real part  

        """

        self._return_types = ('REAL','IMAGINARY','ANGLE','ABSOLUTE','COMPLEX')

        self._dimesion = dimesion
        if return_type in self._return_types:
            self._RETURN_TYPE =return_type

        else:
            self._RETURN_TYPE = 'ABSOLUTE'

    def process(self, process_img):
        """TODO: Docstring for process.

        Args: 
            - process_img: the data to transfer 

        Return:
            - self._transfer_data: the data after transfer
        """

        if self._dimesion == 1:
            tmp_fft = fft.fft(process_img)

        elif self._dimesion == 2:
            tmp_fft = fft.fft2(process_img)

        else:
            process_img = torch.tensor(process_img)
            process_img = torch.unsqueeze(process_img,0)
            tmp_fft = torch.squeeze(fft.fftn(process_img,dim=self._dimesion))

        #self._return_types = ('REAL','IMAGINARY','ANGLE','ABSOLUTE','COMPLEX')
        if self._RETURN_TYPE == 'REAL' :
            return torch.real(tmp_fft)
        elif self._RETURN_TYPE == 'IMAGINARY' :
            return torch.imag(tmp_fft)
        elif self._RETURN_TYPE == 'ANGLE':
            return torch.angle(tmp_fft)
        elif self._RETURN_TYPE == 'ABSOLUTE':
            return torch.absolute(tmp_fft)
        elif self._RETURN_TYPE == 'COMPLEX' :
            return tmp_fft

        return tmp_fft

        
class DiscreteWaveletTransform_(TransformMethod):
    def __init__(self,wavelet='db32'):
        self._wavelet = wavelet

    def process(self,process_img):
        LL, (LH, HL, HH) = pywt.dwt2(process_img, self._wavelet)
        return LL, (LH, HL,HH) 

class DiscreteWaveletTransform(TransformMethod):

    def __init__(self,wavelet='haar',times = 1):
        self._wavelet = wavelet
        self._times = times

    def process(self,process_img):
        #LL = process_img[:,:,0]

        #for i in range(self._times):
        #    LL, (LH, HL, HH) = pywt.dwt2(LL, self._wavelet)


        #return LL

        LL0 = process_img[:,:,0]
        LL1 = process_img[:,:,1]
        LL2 = process_img[:,:,2]

        for i in range(self._times):
            LL0, (LH, HL, HH) = pywt.dwt2(LL0, self._wavelet)
            LL1, (LH, HL, HH) = pywt.dwt2(LL1, self._wavelet)
            LL2, (LH, HL, HH) = pywt.dwt2(LL2, self._wavelet)

        outcome = np.array([LL0,LL1,LL2])
        return outcome


