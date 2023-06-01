import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torch.utils.data import random_split
from torchvision import datasets 
from torchvision import transforms 
import numpy as np

class print_models():
    def __init__(self):
        print('ori_DNN, ori_CNN, ori2_CNN, CIFAR10_CNN,', 
              'AlexNet, VGG16, VGG16_2, ResNet')

class ori_DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear1 = nn.Linear(784, 128) 
        self.batch_norm1 = nn.BatchNorm1d(128) # 배치 정규화
        self.hidden_linear2 = nn.Linear(128, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.ouput_linear = nn.Linear(64, 10)
        nn.init.kaiming_normal_(self.hidden_linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.hidden_linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.ouput_linear.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.hidden_linear1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3)
        x = self.hidden_linear2(x)
        x = self.batch_norm2(x) 
        x = F.relu(x)  
        x = F.dropout(x, 0.2)
        x = self.ouput_linear(x)    
        return x
    
class ori_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential( # batch_size x 32 x 14 x 14
                                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block2 = nn.Sequential( # batch_size x 64 x 7 x 7
                                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.linear1 = nn.Linear(64*7*7, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_block1(x) # batch_size x 32 x 14 x 14
        x = self.conv_block2(x) # batch_size x 64 x 7 x 7
        x = x.view(-1, 64*7*7) # flatten
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class ori2_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential( # batch_size x 32 x 14 x 14
                                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block2 = nn.Sequential( # batch_size x 64 x 7 x 7
                                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                nn.BatchNorm2d(64),
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block3 = nn.Sequential( # batch_size x 128 x 3 x 3
                                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                nn.BatchNorm2d(128),
                                nn.Dropout(0.4),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.linear1 = nn.Linear(128*3*3, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_block1(x) # batch_size x 32 x 14 x 14
        x = self.conv_block2(x) # batch_size x 64 x 7 x 7
        x = self.conv_block3(x) # batch_size x 64 x 7 x 7
        x = x.view(-1, 128*3*3) # flatten
        x = self.linear1(x)
        x = self.batch_norm(x)
        x = F.dropout(x, 0.5)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential( # batch_size x 32 x 16 x 16
                                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block2 = nn.Sequential( # batch_size x 64 x 8 x 8
                                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                nn.BatchNorm2d(64),
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block3 = nn.Sequential( # batch_size x 128 x 4 x 4 
                                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                nn.BatchNorm2d(128),
                                nn.Dropout(0.4),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))  
        self.linear1 = nn.Linear(128*4*4, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_block1(x) # batch_size x 32 x 16 x 16
        x = self.conv_block2(x) # batch_size x 64 x 8 x 8
        x = self.conv_block3(x) # batch_size x 128 x 4 x 4
        
        # reshape할 형상 : (batch_size x 128*4*4)
        # x = x.view(-1, 128*4*4) # option 1 : view
        x = torch.flatten(x, 1) # option 2 : flatten 
        # x = x.reshape(x.shape[0], -1) # option 3 : reshape

        x = self.linear1(x)
        x = self.batch_norm(x)
        x = F.dropout(x, 0.5)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential( # [16, 96, 111, 111]
                                nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(96),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2)) 
        self.conv_block2 = nn.Sequential( # [16, 256, 55, 55]
                                nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(256),                                      
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2)) 
        self.conv_block3 = nn.Sequential( # [16, 384, 55, 55]
                                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(384), 
                                nn.Dropout(0.1),                                    
                                nn.ReLU())      
        self.conv_block4 = nn.Sequential( # [16, 384, 55, 55]   
                                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(384),  
                                nn.Dropout(0.3),                                    
                                nn.ReLU()) 
        self.conv_block5 = nn.Sequential( # [16, 256, 27, 27]
                                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(256),
                                nn.Dropout(0.1), 
                                nn.ReLU(),   
                                nn.MaxPool2d(kernel_size=3, stride=2)) 
        self.linear1 = nn.Linear(256*27*27, 512)
        self.batch_norm = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv_block1(x) 
        x = self.conv_block2(x) 
        x = self.conv_block3(x) 
        x = self.conv_block4(x) 
        x = self.conv_block5(x) 
        x = torch.flatten(x, 1) # reshape: batch_s*256*27*27
        x = F.dropout(x, 0.3)
        x = self.linear1(x)
        x = self.batch_norm(x)
        x = F.dropout(x, 0.1)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential( # [16, 64, 112, 112]
                                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block2 = nn.Sequential( # [16, 128, 56, 56]
                                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block3 = nn.Sequential( # [16, 256, 28, 28] 
                                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(256),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block4 = nn.Sequential( # [16, 512, 14, 14]   
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),  
                                nn.Dropout(0.3),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block5 = nn.Sequential( # [16, 512, 7, 7]
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.Dropout(0.1), 
                                nn.ReLU(), 
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(), 
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),   
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.classifier = nn.Sequential(
                                nn.Linear(512*7*7, 1024),
                                nn.BatchNorm1d(1024),
                                nn.ReLU(),
                                nn.Linear(1024, 64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(64, 2))
    def forward(self, x):
        x = self.conv_block1(x) 
        x = self.conv_block2(x) 
        x = self.conv_block3(x) 
        x = self.conv_block4(x) 
        x = self.conv_block5(x) 
        x = torch.flatten(x, 1) # reshape: batch_s*125*7*7
        x = self.classifier(x)
        return x
    
class VGG16_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential( # [16, 64, 112, 112]
                                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block2 = nn.Sequential( # [16, 128, 56, 56]
                                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block3 = nn.Sequential( # [16, 256, 28, 28] 
                                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(256),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block4 = nn.Sequential( # [16, 512, 14, 14]   
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),  
                                nn.Dropout(0.3),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.conv_block5 = nn.Sequential( # [16, 512, 7, 7]
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.Dropout(0.1), 
                                nn.ReLU(), 
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(), 
                                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),   
                                nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # [16, 512, 1, 1]
        self.classifier = nn.Sequential(
                                nn.Linear(512, 64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(64, 2))
    def forward(self, x):
        x = self.conv_block1(x) 
        x = self.conv_block2(x) 
        x = self.conv_block3(x) 
        x = self.conv_block4(x) 
        x = self.conv_block5(x) 
        x = self.avg_pool(x)
        x = torch.flatten(x, 1) # reshape: batch_s*512
        x = self.classifier(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential( # [16, 64, 56, 56] 
                            # BatchNorm 계층은 편향값의 효과를 보완해주므로 관례상 생략
                            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),                        
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) 
        self.shortcut2 = nn.Sequential(
                            nn.Conv2d(64, 256, kernel_size=1, stride=1), 
                            nn.BatchNorm2d(256))
        self.conv2_x = nn.Sequential( # [16, 256, 56, 56]
                            ResBlock(64, 64, shortcut=self.shortcut2, stride=1),                                 
                            ResBlock(256, 64, shortcut=None, stride=1),
                            ResBlock(256, 64, shortcut=None, stride=1)) 
        self.shortcut3 = nn.Sequential(
                            nn.Conv2d(256, 512, kernel_size=1, stride=2), 
                            nn.BatchNorm2d(512))      
        self.conv3_x = nn.Sequential( # [16, 512, 28, 28]   
                            ResBlock(256, 128, shortcut=self.shortcut3, stride=2),
                            ResBlock(512, 128, shortcut=None, stride=1),
                            ResBlock(512, 128, shortcut=None, stride=1),
                            ResBlock(512, 128, shortcut=None, stride=1)) 
        self.shortcut4 = nn.Sequential(
                            nn.Conv2d(512, 1024, kernel_size=1, stride=2), 
                            nn.BatchNorm2d(1024))      
        self.conv4_x = nn.Sequential( # [16, 1024, 14, 14] 
                            ResBlock(512, 256, shortcut=self.shortcut4, stride=2),
                            ResBlock(1024, 256, shortcut=None, stride=1),
                            ResBlock(1024, 256, shortcut=None, stride=1),
                            ResBlock(1024, 256, shortcut=None, stride=1),
                            ResBlock(1024, 256, shortcut=None, stride=1),
                            ResBlock(1024, 256, shortcut=None, stride=1)) 
        self.shortcut5 = nn.Sequential(
                            nn.Conv2d(1024, 2048, kernel_size=1, stride=2), 
                            nn.BatchNorm2d(2048))    
        self.conv5_x = nn.Sequential( # [16, 2048, 7, 7]  
                            ResBlock(1024, 512, shortcut=self.shortcut5, stride=2),
                            ResBlock(2048, 512, shortcut=None, stride=1),
                            ResBlock(2048, 512, shortcut=None, stride=1)) 
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # [16, 2048, 1, 1]
        self.classifier = nn.Sequential(
                            nn.Linear(2048, 2),
                            # nn.BatchNorm1d(64),
                            # nn.ReLU(),
                            # nn.Linear(64, 2)
                            )

    def forward(self, x):
        x = self.conv1(x) # [16, 64, 56, 56]
        x = self.conv2_x(x) # [16, 256, 56, 56]
        x = self.conv3_x(x) # [16, 512, 28, 28] 
        x = self.conv4_x(x) # [16, 1024, 14, 14] 
        x = self.conv5_x(x) # [16, 2048, 7, 7] 
        x = self.avg_pool(x) # [16, 2048, 1, 1] 
        x = torch.flatten(x, 1) # (batch_size x 2048)
        x = self.classifier(x)    
        return x