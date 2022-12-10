import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch

from src.model.cnn_model import DwtConvNet4Class
from torchsummary import summary

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dwt_cnn = DwtConvNet4Class().to(device)
    summary(dwt_cnn,(3,26,22))
