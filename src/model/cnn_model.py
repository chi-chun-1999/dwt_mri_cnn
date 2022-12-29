import torch
from torch import nn
import torch.nn.functional as F


class DwtConvNet4Class(nn.Module):
    def __init__(self):
        super(DwtConvNet4Class,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.batch_normal1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.droup_out = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(in_channels=24,out_channels=48,kernel_size=3,stride=1,padding=1)
        self.batch_normal2 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)


        self.conv3 = nn.Conv2d(in_channels=48,out_channels=96,kernel_size=3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.batch_normal3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(in_channels=96,out_channels=192,kernel_size=3,stride=1,padding=1)
        self.batch_normal4 = nn.BatchNorm2d(192)

        self.fc1 = nn.Linear(5760,512)
        self.fc2 = nn.Linear(512,1024)

        self.fc_final = nn.Linear(1024,4)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batch_normal1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.droup_out(x)

        x = self.conv2(x)
        x = self.batch_normal2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.droup_out(x)

        x = self.conv3(x)
        x = self.batch_normal3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.droup_out(x)

        x = self.conv4(x)
        x = self.batch_normal4(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.droup_out(x)

        x = torch.flatten(x,1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.sigmoid(self.fc_final(x))

        return x

class DwtConvNet2Class(nn.Module):
    def __init__(self):
        super(DwtConvNet2Class,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.batch_normal1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.droup_out = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(in_channels=24,out_channels=48,kernel_size=3,stride=1,padding=1)
        self.batch_normal2 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)


        self.conv3 = nn.Conv2d(in_channels=48,out_channels=96,kernel_size=3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.batch_normal3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(in_channels=96,out_channels=192,kernel_size=3,stride=1,padding=1)
        self.batch_normal4 = nn.BatchNorm2d(192)

        #self.fc1 = nn.Linear(5760,512)
        #self.fc1 = nn.Linear(3072,512)
        self.fc1 = nn.Linear(192,512)
        #self.fc1 = nn.Linear(1152,512)
        self.fc2 = nn.Linear(512,1024)

        self.fc_final = nn.Linear(1024,2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batch_normal1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.droup_out(x)

        x = self.conv2(x)
        x = self.batch_normal2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.droup_out(x)

        x = self.conv3(x)
        x = self.batch_normal3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.droup_out(x)

        x = self.conv4(x)
        x = self.batch_normal4(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.droup_out(x)

        x = torch.flatten(x,1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.sigmoid(self.fc_final(x))

        return x

class DwtConvNet2Class5Block(nn.Module):
    def __init__(self):
        super(DwtConvNet2Class5Block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.batch_normal1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.droup_out = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(in_channels=24,out_channels=48,kernel_size=3,stride=1,padding=1)
        self.batch_normal2 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)


        self.conv3 = nn.Conv2d(in_channels=48,out_channels=96,kernel_size=3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.batch_normal3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(in_channels=96,out_channels=192,kernel_size=3,stride=1,padding=1)
        self.batch_normal4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(in_channels=192,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.batch_normal5 = nn.BatchNorm2d(384)

        #self.fc1 = nn.Linear(5760,512)
        #self.fc1 = nn.Linear(3072,512)
        self.fc1 = nn.Linear(1536,512)
        #self.fc1 = nn.Linear(192,512)
        #self.fc1 = nn.Linear(1152,512)
        self.fc2 = nn.Linear(512,1024)

        self.fc_final = nn.Linear(1024,2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batch_normal1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.droup_out(x)

        x = self.conv2(x)
        x = self.batch_normal2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.droup_out(x)

        x = self.conv3(x)
        x = self.batch_normal3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.droup_out(x)

        x = self.conv4(x)
        x = self.batch_normal4(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.droup_out(x)

        x = self.conv5(x)
        x = self.batch_normal5(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.droup_out(x)

        x = torch.flatten(x,1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.sigmoid(self.fc_final(x))

        return x

class DwtConvNet2Class3Block(nn.Module):
    def __init__(self):
        super(DwtConvNet2Class3Block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.batch_normal1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.droup_out = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(in_channels=24,out_channels=48,kernel_size=3,stride=1,padding=1)
        self.batch_normal2 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)


        self.conv3 = nn.Conv2d(in_channels=48,out_channels=96,kernel_size=3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.batch_normal3 = nn.BatchNorm2d(96)

        #self.conv4 = nn.Conv2d(in_channels=96,out_channels=192,kernel_size=3,stride=1,padding=1)
        #self.batch_normal4 = nn.BatchNorm2d(192)

        #self.fc1 = nn.Linear(576,512)
        #self.fc1 = nn.Linear(192,512)
        self.fc1 = nn.Linear(6144,512)
        self.fc2 = nn.Linear(512,1024)

        self.fc_final = nn.Linear(1024,2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batch_normal1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.droup_out(x)

        x = self.conv2(x)
        x = self.batch_normal2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.droup_out(x)

        x = self.conv3(x)
        x = self.batch_normal3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.droup_out(x)

        x = torch.flatten(x,1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.sigmoid(self.fc_final(x))

        return x


class DwtConvNet2Class2Block(nn.Module):
    def __init__(self):
        super(DwtConvNet2Class2Block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.batch_normal1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.droup_out = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(in_channels=24,out_channels=48,kernel_size=3,stride=1,padding=1)
        self.batch_normal2 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)


        self.conv3 = nn.Conv2d(in_channels=48,out_channels=96,kernel_size=3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.batch_normal3 = nn.BatchNorm2d(96)

        #self.conv4 = nn.Conv2d(in_channels=96,out_channels=192,kernel_size=3,stride=1,padding=1)
        #self.batch_normal4 = nn.BatchNorm2d(192)

        self.fc1 = nn.Linear(1440,512)
        #self.fc1 = nn.Linear(192,512)
        #self.fc1 = nn.Linear(1152,512)
        self.fc2 = nn.Linear(512,1024)

        self.fc_final = nn.Linear(1024,2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batch_normal1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.droup_out(x)

        x = self.conv2(x)
        x = self.batch_normal2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.droup_out(x)

        x = torch.flatten(x,1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.sigmoid(self.fc_final(x))

        return x


class EasyConvNet(nn.Module):
    def __init__(self,first_dense_layer_par=1296):
        super(EasyConvNet, self).__init__()
        # image shape is 1 * 28 * 28, where 1 is one color channel
        # 28 * 28 is the image size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)    # output shape = 3 * 24 * 24
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                       # output shape = 3 * 12 * 12
        # intput shape is 3 * 12 * 12
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5)    # output shape = 9 * 8 * 8
        # add another max pooling, output shape = 9 * 4 * 4
        #self.fc1 = nn.Linear(9*4*4, 100)
        self.fc1 = nn.Linear(first_dense_layer_par, 64)
        self.fc2 = nn.Linear(64, 32)
        # last fully connected layer output should be same as classes
        self.fc3 = nn.Linear(32, 4)

        
    def forward(self, x):
        # first conv
        x = self.pool(F.relu(self.conv1(x)))
        # second conv
        x = self.pool(F.relu(self.conv2(x)))
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        #print(x.view(x.size(0), -1).shape)
        # fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


