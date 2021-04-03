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
import time
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from soft_n_cut_loss import soft_n_cut_loss

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
                                            [1,  0,  -1]]]])).float().cuda(), requires_grad=False)

horizontal_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,   1,  1], 
                                              [0,   0,  0], 
                                              [-1 ,-1, -1]]]])).float().cuda(), requires_grad=False)

    
def train_op(model, optimizer, input, psi=0.5):
    enc = model(input, returns='enc') # The output of the UEnc is a normalized 224 × 224 × K dense prediction.
    n_cut_loss=soft_n_cut_loss(input, enc)
    n_cut_loss.backward() 
    optimizer.step()
    optimizer.zero_grad()
    dec = model(input, returns='dec')
    rec_loss=reconstruction_loss(input, dec)
    rec_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return (model, n_cut_loss, rec_loss)

def reconstruction_loss(x, x_prime):
    # binary_cross_entropy = F.binary_cross_entropy(x_prime, x, reduction='sum')
    # return binary_cross_entropy
    criterionIdt = torch.nn.L1Loss() #prob l2 or mseless here
    rec_loss = criterionIdt(x_prime, x)
    return rec_loss

def test():
    wnet=WNet.WNet(4)
    synthetic_data=torch.rand((1, 3, 128, 128))
    optimizer=torch.optim.SGD(wnet.parameters(), 0.001).cuda()
    train_op(wnet, optimizer, synthetic_data)

def show_image(image):
    img = image.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

def main():
    args, unknown = parser.parse_known_args()

    n_cut_losses_avg = []
    rec_losses_avg = []
    k = args.squeeze
    wnet = WNet.WNet(k)
    wnet = wnet.cuda()
    learning_rate = 0.003
    optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)
    # transforms.CenterCrop(224),
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.input_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
                                          
    
    for epoch in range(args.epochs):
        if (epoch % 1000 == 0):
            learning_rate = learning_rate/10
            optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)
        print("Epoch = " + str(epoch))

        n_cut_losses = []
        rec_losses = []
        start_time = time.time()
        for (idx, batch) in enumerate(dataloader):
            if(idx > 50): break
            wnet, n_cut_loss, rec_loss = train_op(wnet, optimizer, batch[0].cuda())
            n_cut_losses.append(n_cut_loss.detach())
            rec_losses.append(rec_loss.detach())
        n_cut_losses_avg.append(torch.mean(torch.FloatTensor(n_cut_losses)))
        rec_losses_avg.append(torch.mean(torch.FloatTensor(rec_losses)))
        print("--- %s seconds ---" % (time.time() - start_time))


    images, labels = next(iter(dataloader))
    enc, dec = wnet(images.cuda())
    # print(images.shape)
    # print(enc.shape)
    # print(dec.shape)

    torch.save(wnet.state_dict(), "model")
    np.save("rec_losses", n_cut_losses_avg)
    np.save("n_cut_losses", rec_losses_avg)
    print("Done")

if __name__ == '__main__':
    main()


# python .\train.py --e 100 --input_folder="data/images/" --output_folder="/output/"