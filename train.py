# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:38:02 2018
@author: Tao Lin

Implementation of the W-Net unsupervised image segmentation architecture
"""

import argparse
import torch.nn as nn
import numpy as np
import time
import datetime
import torch
from torchvision import datasets, transforms
from utils.soft_n_cut_loss import soft_n_cut_loss

import WNet
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with WNet')
parser.add_argument('--name', metavar='name', default=str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')), type=str,
                    help='Name of model')
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


def train_op(model, optimizer, input, k, img_shape, psi=0.5):
    softmax = nn.Softmax2d()
    enc = model(input, returns='enc') # The output of the UEnc is a normalized 224 × 224 × K dense prediction.
    n_cut_loss=soft_n_cut_loss(input, softmax(enc), k, img_shape)
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
    #m = nn.Sigmoid()
    criterionIdt = torch.nn.L1Loss() #prob l2 or mseless here
    rec_loss = criterionIdt(x_prime, x)
    return rec_loss

def test():
    wnet=WNet.WNet(4)
    synthetic_data=torch.rand((1, 3, 128, 128))
    optimizer=torch.optim.SGD(wnet.parameters(), 0.001) #.cuda()
    train_op(wnet, optimizer, synthetic_data)

def show_image(image):
    img = image.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

def main():
    # Load the arguments
    args, unknown = parser.parse_known_args()

    # Check if CUDA is available
    CUDA = torch.cuda.is_available()

    # Create empty lists for average N_cut losses and reconstruction losses
    n_cut_losses_avg = []
    rec_losses_avg = []

    # Squeeze k
    k = args.squeeze
    img_size = (64, 64)
    wnet = WNet.WNet(k)
    if(CUDA):
        wnet = wnet.cuda()
    learning_rate = 0.03
    optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)
    # transforms.CenterCrop(224),
    transform = transforms.Compose([transforms.Resize(img_size),
                                transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.input_folder, transform=transform)

    # Train 1 image set batch size=1 and set shuffle to False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Run for every epoch
    for epoch in range(args.epochs):

        # At 1000 epochs divide SGD learning rate by 10
        if (epoch % 1000 == 0):
            learning_rate = learning_rate/10
            optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)

        # Print out every epoch:
        print("Epoch = " + str(epoch))

        # Create empty lists for N_cut losses and reconstruction losses
        n_cut_losses = []
        rec_losses = []
        start_time = time.time()

        for (idx, batch) in enumerate(dataloader):
            # Train 1 image idx > 1
            # if(idx > 1): break

            # Train Wnet with CUDA if available
            if CUDA:
                batch[0] = batch[0].cuda()
            
            wnet, n_cut_loss, rec_loss = train_op(wnet, optimizer, batch[0].cuda(), k, img_size)

            n_cut_losses.append(n_cut_loss.detach())
            rec_losses.append(rec_loss.detach())

        n_cut_losses_avg.append(torch.mean(torch.FloatTensor(n_cut_losses)))
        rec_losses_avg.append(torch.mean(torch.FloatTensor(rec_losses)))
        print("--- %s seconds ---" % (time.time() - start_time))


    images, labels = next(iter(dataloader))

    # Run wnet with cuda if enabled
    if CUDA:
        images = images.cuda()

    enc, dec = wnet(images)

    torch.save(wnet.state_dict(), "model_" + args.name)
    np.save("losses_output/n_cut_losses_" + args.name, n_cut_losses_avg)
    np.save("losses_output/rec_losses_" + args.name, rec_losses_avg)
    print("Done")

if __name__ == '__main__':
    main()


# python .\train.py --e 100 --input_folder="data/images/" --output_folder="/output/"