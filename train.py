# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:32:16 2018
@author: Tao Lin

Training and Predicting with the W-Net unsupervised segmentation architecture
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

import WNet
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with WNet')
parser.add_argument('--in_Chans', metavar='C', default=3, type=int, 
                    help='number of input channels')
parser.add_argument('--squeeze', metavar='K', default=4, type=int, 
                    help='Depth of squeeze layer')
parser.add_argument('--out_Chans', metavar='O', default=3, type=int, 
                    help='Output Channels')
parser.add_argument('--epochs', metavar='e', default=100, type=int, 
                    help='epochs')
parser.add_argument('--input_folder', metavar='f', default=None, type=str, 
                    help='Folder of input images')
parser.add_argument('--output_folder', metavar='of', default=None, type=str, 
                    help='folder of output images')

vertical_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,  0,  -1], 
                                            [1,  0,  -1], 
                                            [1,  0,  -1]]]])).float(), requires_grad=False)

horizontal_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,   1,  1], 
                                              [0,   0,  0], 
                                              [-1 ,-1, -1]]]])).float(), requires_grad=False)

def gradient_regularization(softmax, device='cuda'):
    vert=torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), vertical_sobel) for i in range(softmax.shape[1])], 1)
    hori=torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), horizontal_sobel) for i in range(softmax.shape[1])], 1)
    # print('vert', torch.sum(vert))
    # print('hori', torch.sum(hori))
    mag=torch.pow(torch.pow(vert, 2)+torch.pow(hori, 2), 0.5)
    mean=torch.mean(mag)
    return mean
    
def train_op(model, optimizer, input, psi=0.5):
    enc = model(input, returns='enc') # The output of the UEnc is a normalized 224 × 224 × K dense prediction.
    n_cut_loss=gradient_regularization(enc)*psi
    n_cut_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    dec = model(input, returns='dec')
    rec_loss=torch.mean(torch.pow(torch.pow(input, 2) + torch.pow(dec, 2), 0.5))*(1-psi)
    rec_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return model

def test():
    wnet=WNet.WNet(4)
    synthetic_data=torch.rand((1, 3, 128, 128))
    optimizer=torch.optim.SGD(wnet.parameters(), 0.001)
    train_op(wnet, optimizer, synthetic_data)

def show_image(image):
    img = image.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

def main():
    args = parser.parse_args()


    # Works with squeeze = 128,
    wnet = WNet.WNet(args.squeeze)
    learning_rate = 0.003
    dropout = 0.65
    optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)
    
    # transforms.CenterCrop(224),
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.input_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    images, labels = next(iter(dataloader))
    # show_image(images[0])
    
    # iterate thru training data e times
    for epoch in range(args.epochs):
        if (epoch % 1000 == 0):
            learning_rate = learning_rate/10
        print("Epoch = " + str(epoch))
        for (idx, batch) in enumerate(dataloader):
            if( idx >= 50): break
            print(idx)
            # batch consists of images and labels.
            wnet = train_op(wnet, optimizer, batch[0], dropout)

    images, labels = next(iter(dataloader))
    enc, dec = wnet(images)
    # print(images.shape)
    # print(enc.shape)
    # print(dec.shape)
    show_image(images[0])
    show_image(enc[0, :, :, :].detach())
    show_image(dec[0, :, :, :].detach())


    torch.save(wnet.state_dict(), "model")

if __name__ == '__main__':
    main()


# python .\train.py --e 100 --input_folder="data/images/train" --output_folder="/output/"