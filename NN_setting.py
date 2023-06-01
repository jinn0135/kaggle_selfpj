import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader 
from torch.utils.data import random_split
from torchvision import datasets 
from torchvision import transforms 
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class setting():
    def __init__(self):
        self.trainset, self.testset = None, None
        self.train_set, self.valid_set = None, None

    def getDataset_input(self, trainset, testset):
        self.trainset, self.testset = trainset, testset
        print('trainset length:', len(self.trainset), '/ testset length:', len(self.testset))

    def getDataset(self, getdata, dir, dircts, transform):
        if len(dircts)==2: 
            self.trainset = getdata(dir+dircts[0], transform)
            self.testset = getdata(dir+dircts[1], transform)
        else:
            trainset = getdata(dir+dircts[0], transform)
            train_idxs, test_idxs, _, _ = train_test_split(
                range(len(trainset)), trainset.targets, test_size=0.2,
                stratify=trainset.targets, random_state=42)
            self.trainset = Subset(trainset, train_idxs)
            self.testset = Subset(trainset, test_idxs)
        print('trainset length:', len(self.trainset), '/ testset length', len(self.testset))

    def getValid(self, valid_s=0.2, classify=True):
        if classify:
            train_idxs, valid_idxs, _, _, = train_test_split(
                    range(len(self.trainset)), self.trainset.targets, test_size=valid_s,
                    stratify=self.trainset.targets, random_state=42)
        else:
            train_idxs, valid_idxs = train_test_split(
                    range(len(self.trainset)), test_size=valid_s, random_state=42)
        print('trainset length:', len(train_idxs), '/ validset length:', len(valid_idxs))
        self.train_set = Subset(self.trainset, train_idxs)
        self.valid_set = Subset(self.trainset, valid_idxs)
        if classify: print(self.train_set[0][0].size(), self.train_set[0][1])
        else: print(self.train_set[0][0].shape, self.train_set[0][1].shape)
    
    def getDataloader(self, batch_s=16):
        trainloader = DataLoader(self.train_set, batch_size=batch_s, shuffle=True)
        validloader = DataLoader(self.valid_set, batch_size=batch_s, shuffle=True)
        testloader = DataLoader(self.testset, batch_size=batch_s, shuffle=True)
        print('train, valid, test:', len(trainloader), len(validloader), len(testloader))
        return trainloader, validloader, testloader

    def showimg(self, labels_map, data):
        fig, ax = plt.subplots(4,8, figsize=(14,8))
        ax = ax.flatten()
        for i in range(32):
            item = data[np.random.randint(0, len(data))]
            img, label = item[0].permute(1,2,0), item[1]
            ax[i].axis('off'); ax[i].imshow(img)
            ax[i].set_title(labels_map[label])

    def showtransimg(self, img):
        fig, ax = plt.subplots(1,5, figsize=(10,3))
        for i, trans in enumerate(['ori','gray','rotate','crop','hori']):
            trans_img = self.transimg(img, trans)
            if trans=='gray': ax[i].imshow(trans_img.squeeze(), cmap='gray')
            else: ax[i].imshow(trans_img.permute(1,2,0))
            ax[i].set_title(trans)
            ax[i].axis('off')

    def transimg(self, img, trans):
        if trans=='ori': return img
        elif trans=='gray': return transforms.Grayscale()(img)
        elif trans=='rotate': return transforms.RandomRotation(degrees=(0,180))(img)
        elif trans=='crop': return transforms.RandomCrop(size=(128,128))(img)
        elif trans=='hori': return transforms.RandomHorizontalFlip(p=0.3)(img)
        elif trans=='compose':
            return transforms.Compose([
                                transforms.RandomCrop(size=(128, 128)),
                                transforms.RandomRotation(degrees=(0, 180)),
                                transforms.Resize((400, 224)),
                                transforms.ToTensor()])

    
from torch.utils.data import Dataset
import glob
from PIL import Image # Image.open(path)
import cv2
import os
import pandas as pd
import albumentations as A # trochvision transforms 보다 빠름(label도 같이 변환가능)
from albumentations.pytorch import ToTensorV2
class createDataset(Dataset):
    def __init__(self, root, transform, using='torchvision', label_csv=None):
        self.label_csv = label_csv
        if self.label_csv is None: 
            self.filepaths = glob.glob(root+'*/*.jpg')
        else: 
            self.root = root
            self.key_pts_frame = pd.read_csv(label_csv)
        self.transform = transform
        self.using = using

    def __len__(self):
        if self.label_csv is None: return len(self.filepaths)
        else: return len(self.key_pts_frame)

    def __getitem__(self, idx):
        if self.label_csv is None: img_filepath = self.filepaths[idx]
        else: img_filepath = os.path.join(self.root, self.key_pts_frame.iloc[idx,0])
        if self.using=='A': 
            img = cv2.imread(img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.label_csv is None: 
                transformed_img = self.transform(image=img)['image']
        else: 
            img = Image.open(img_filepath)
            transformed_img = self.transform(img)

        if self.label_csv==None:
            dir_label = img_filepath.split('/')[-2]
            if dir_label=='cats': label = 0
            else: label = 1
            return transformed_img, label
        else:
            keypoints = self.key_pts_frame.iloc[idx,1:].values.astype('float').reshape(-1,2)
            transformed = self.transform(image=img, keypoints=keypoints)
            transformed['keypoints'] = torch.tensor(transformed['keypoints']).flatten()
            # transformed['keypoints'] = (transformed['keypoints']-100)/50
            return transformed['image'], transformed['keypoints']
    
import cv2
class Transimg():
    def __init__(self):
        pass
    def showimg(self, img, w,h):
        trans_li = ['resize','bright','veri_flip','hori_flip','shiftrotate',
                    'blur','oneof','norm']
        fig, ax = plt.subplots(len(trans_li),5, figsize=(w,h), constrained_layout=True)
        for i, aug in enumerate(trans_li):
            ax[i][0].imshow(img)
            ax[i][0].set_title('original')
            ax[i][0].axis('off')
            for j in range(1,5):
                augment_img = self.augmentor(aug)(image=img)['image']
                ax[i][j].imshow(augment_img)
                ax[i][j].set_title(aug)
                ax[i][j].axis('off')

    def augmentor(self, aug):
        if aug=='resize': return A.Resize(450,650)
        elif aug=='bright': return A.RandomBrightnessContrast(
                                    brightness_limit=0.2, contrast_limit=0.2, p=0.3)
        elif aug=='veri_flip': return A.VerticalFlip(p=0.2)
        elif aug=='hori_flip': return A.HorizontalFlip(p=0.4)
        elif aug=='shiftrotate': return A.ShiftScaleRotate(
                            shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.3)
        elif aug=='blur': return A.Blur(blur_limit=(7,10), p=1)
        elif aug=='oneof': 
            return A.OneOf([
                A.VerticalFlip(p=1), A.HorizontalFlip(p=1),
                A.Rotate(limit=(45, 90), p=1, border_mode=cv2.BORDER_CONSTANT)], p=0.8)
        elif aug=='norm': return A.Normalize()


# transform = transforms.Compose([myTrans1(), # Resize
#                                 myTrans2()]) # ToTensor
class myTrans1(): # Resize
    def __init__(self, output_s):
        self.output_s = output_s
    def __call__(self, input):
        return cv2.resize(np.array(input), (self.output_s[1], self.output_s[0]))
class myTrans2(): # ToTensor
    def __init__(self):
        pass
    def __call__(self, input):
        tensor_input = torch.FloatTensor(np.array(input)/255)
        return tensor_input.permute(2,0,1)
