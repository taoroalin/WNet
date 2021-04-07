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
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from utils.crf import dense_crf
from cv2 import imread, imwrite, resize

import WNet
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with WNet')
parser.add_argument('--model', metavar='C', default="model", type=str,
                    help='name of the saved model')
parser.add_argument('--image', metavar='C', default=None, type=str,
                    help='path to the image')
parser.add_argument('--squeeze', metavar='K', default=4, type=int,
                    help='Depth of squeeze layer')

def show_image(image):
    img = image.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

def main():
    args = parser.parse_args()
    model = WNet.WNet(args.squeeze)

    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor()])

    img = Image.open("data2/images/train/1head.png").convert('RGB')
    x = transform(img)[None, :, :, :]

    enc, dec = model(x)
    show_image(x[0])
    # TODO: torch sum/ stack?
    show_image(enc[0, :3, :, :].detach())
    # show_image(torch.argmax(enc[:,:,:,:], dim=1))
    # show_image(dec[0, :, :, :].detach())
    # now put enc in crf
    segment = enc[0, :, :, :].detach()
    # put in tensor here?

    orimg = imread("data2/images/train/1head.png")
    img = resize(orimg, (64, 64))
    Q = dense_crf(img, segment.numpy())

    print(type(Q))
    Q = np.argmax(Q, axis=0)
    print(len(Q))

    print(np.unique(Q))
    plt.imshow(Q)
    plt.show()

if __name__ == '__main__':
    main()

# python .\predict.py --image="data/images/test/2018.jpg"