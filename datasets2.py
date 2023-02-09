from skimage import io
import os.path

import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms as transforms
from model import GeneratorResNet, Discriminator



class Div2k(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform = None, type='train'):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.type = type

    def __len__(self):
        return 800 if self.type == 'train' else 100
    def __getitem__(self, index):
        string = str(index+1) if self.type == 'train' else str(index + 1 + 800)
        while(len(string)<4):
            string = '0'+string

        lr_img_path = os.path.join(self.low_res_dir,'X4',string+'x4.png')
        hr_img_path = os.path.join(self.high_res_dir, string+'.png')
        lr_img = io.imread(lr_img_path)
        hr_img = io.imread(hr_img_path)
        if self.transform:
            lr = self.transform(lr_img)
            hr = self.transform(hr_img)

        lr, hr = random_crop(lr_img, hr_img, hr_crop_size=96, scale=4)
        lr, hr = scale(lr, hr)
        return (lr, hr)

class Set14(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform = None, type='train'):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.type = type

    def __len__(self):
        return 13
    def __getitem__(self, index):
        string = str(index+1) if self.type == 'train' else str(index + 1 + 800)
        while(len(string)<3):
            string = '0'+string

        lr_img_path = os.path.join(self.low_res_dir,'LR_4',"img_"+string+'_SRF_4_LR.png')
        hr_img_path = os.path.join(self.high_res_dir, 'HR',"img_"+string+'_SRF_4_HR.png')
        lr_img = io.imread(lr_img_path)
        hr_img = io.imread(hr_img_path)
        if self.transform:
            lr = self.transform(lr_img)
            hr = self.transform(hr_img)

        # lr, hr = random_crop(lr_img, hr_img, hr_crop_size=96, scale=4)
        # lr, hr = scale(lr, hr)
        return (lr, hr)

class Set5(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform = None, type='train'):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.type = type

    def __len__(self):
        return 5
    def __getitem__(self, index):
        string = str(index+1) if self.type == 'train' else str(index + 1 + 800)
        while(len(string)<3):
            string = '0'+string

        lr_img_path = os.path.join(self.low_res_dir,'LR_4',"img_"+string+'_SRF_4_LR.png')
        hr_img_path = os.path.join(self.high_res_dir, 'HR',"img_"+string+'_SRF_4_HR.png')
        lr_img = io.imread(lr_img_path)
        hr_img = io.imread(hr_img_path)
        if self.transform:
            lr = self.transform(lr_img)
            hr = self.transform(hr_img)

        # lr, hr = random_crop(lr_img, hr_img, hr_crop_size=96, scale=4)
        # lr, hr = scale(lr, hr)
        return (lr, hr)

class Flickr2k(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform = None):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform

    def __len__(self):
        return 2650
    def __getitem__(self, index):
        string = str(index+1)
        while(len(string)<4):
            string = '0'+string


        lr_img_path = os.path.join(self.low_res_dir,'X4', '00'+string+'x4.png')
        hr_img_path = os.path.join(self.high_res_dir, '00'+string+'.png')
        lr_img = io.imread(lr_img_path)
        hr_img = io.imread(hr_img_path)
        if self.transform:
            lr = self.transform(lr_img)
            hr = self.transform(hr_img)

        lr, hr = random_crop(lr_img, hr_img, hr_crop_size=96, scale=4)
        lr, hr = scale(lr, hr)
        return (lr, hr)


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
    torch.manual_seed(42)

    lr_crop_size = hr_crop_size // scale
    lr_img_shape = lr_img.shape[:2]

    lr_w = torch.randint(low=0, high=lr_img_shape[1] - lr_crop_size + 1, size=(1,1))
    lr_h = torch.randint(low=0, high=lr_img_shape[0] - lr_crop_size + 1, size=(1,1))


    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = torch.tensor(lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size])
    hr_img_cropped = torch.tensor(hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size])

    return lr_img_cropped, hr_img_cropped

def scale(lr_img, hr_img):
    lr_img = torch.div(
        torch.subtract(
            lr_img,
            torch.min(lr_img)
        ),
        torch.subtract(
            torch.max(lr_img),
            torch.min(lr_img)
        )
    )
    hr_img = torch.div(
        torch.subtract(
            hr_img,
            torch.min(hr_img)
        ) ,
        torch.subtract(
            torch.max(hr_img),
            torch.min(hr_img)
        )
    )
    hr_img = hr_img * 2 - 1
    return lr_img, hr_img


torch.manual_seed(42)
#div2k train
# div2k = Div2k(low_res_dir='.div2k/images/DIV2K_train_LR_bicubic', high_res_dir='.div2k/images/DIV2K_train_HR', transform=transforms.ToTensor())
# #div2k validation
# div2k_val = Div2k(low_res_dir='.div2k/images/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic', high_res_dir='.div2k/images/DIV2K_valid_HR/DIV2K_valid_HR', transform=transforms.ToTensor(), type='valid')
# train_loader = DataLoader(dataset=div2k, batch_size=16)
# train_loader = DeviceDataLoader(train_loader, 'cuda')

flickr = Flickr2k(low_res_dir='flickr/Flickr2K_LR_bicubic', high_res_dir='flickr/Flickr2K_HR', transform=transforms.ToTensor())
flickr_train, flickr_val = torch.utils.data.random_split(flickr, [int(flickr.__len__()*0.8), int(flickr.__len__()*0.2)])
train_loader = DataLoader(dataset=flickr_train, batch_size=16)
train_loader_test = DataLoader(dataset=flickr_train, batch_size=1)
train_loader = DeviceDataLoader(train_loader, 'cuda')
train_loader_test = DeviceDataLoader(train_loader_test, 'cuda')
val_loader = DataLoader(dataset=flickr_val, batch_size=10)
val_loader = DeviceDataLoader(val_loader, 'cuda')

set14 = Set14(low_res_dir='Set14_SR/Set14', high_res_dir='Set14_SR/Set14', transform=transforms.ToTensor())
set14_loader = DataLoader(dataset=set14, batch_size=1)

set5 = Set5(low_res_dir='Set5_SR/Set5', high_res_dir='Set5_SR/Set5', transform=transforms.ToTensor())
set5_loader = DataLoader(dataset=set5, batch_size=1)



#concatenate training data of Div2k and Flickr2k
# train_dev_sets = torch.utils.data.ConcatDataset([div2k, flickr])
#
# #create data loader for div2k and flickr2k
# train_dev_loader = DataLoader(dataset=train_dev_sets, batch_size=16, shuffle=True)
# train_loader = DeviceDataLoader(train_dev_loader, 'cuda')

#concatenate validation data of Div2k and Flickr2k
# val_dev_sets = torch.utils.data.ConcatDataset([div2k_val, flickr_val])
# print(val_dev_sets.__len__())

#create data loader for div2k and flickr2k validation datasets
# valid_dev_loader = DataLoader(dataset=div2k_val, batch_size=10)
# val_loader = DeviceDataLoader(valid_dev_loader, 'cuda')

print(len(train_loader))
print(torch.cuda.is_available())
# flickr_loader222 = DataLoader(dataset=flickr, batch_size=1)
# gen = GeneratorResNet()
# idx=0
# for lr,hr in flickr_loader222:
#     print(lr.shape)
#     lr = lr.permute((0,3,1,2))
#     print(idx)
#     print(gen(lr)[0][0][0][0].item())
#     idx+=1
#     print('lr.device:', lr.device)
#     print('hr.device:', hr.device)
#     fig2 = plt.figure()
#     for i in range(8):
#         img = lr[i]
#         fig2.add_subplot(2, 4, i + 1)
#         plt.imshow(img)
#     plt.show()
#     fig3 = plt.figure()
#     for i in range(8):
#         img = hr[i]
#         fig3.add_subplot(2, 4, i + 1)
#         plt.imshow(img)
#     plt.show()

