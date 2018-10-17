# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:36:27 2018

@author: tao
"""

import train
import WNet
import torch
import numpy as np

def EncoderTest(verbose=True):
    shape=(2, 4, 224, 224)
    encoder=WNet.UEnc(shape[1])
    data=torch.rand((shape[0], 3, shape[2], shape[3]))
    encoded=encoder(data)
    assert tuple(encoded.shape)==shape
    var=torch.var(encoded)
    mean=torch.mean(encoded)
    if verbose:
        print('Passed Encoder Test with var=%s and mean=%s' % (var, mean))
    return encoded
def DecoderTest():
    shape=(2, 4, 224, 224)
    out_shape=(2, 3, 224, 224)
    decoder=WNet.UDec(shape[1])
    data=torch.rand(tuple(shape))
    decoded=decoder(data)
    assert tuple(decoded.shape)==out_shape
    var=torch.var(decoded)
    mean=torch.mean(decoded)
    print('Passed Decoder Test with var=%s and mean=%s' % (var, mean))
def WNetTest():
    encoded=EncoderTest(verbose=False)
    decoder=WNet.UDec(4)
    reproduced=decoder(encoded)
    var=torch.var(reproduced)
    mean=torch.mean(reproduced)
    print('Passed Decoder Test with var=%s and mean=%s' % (var, mean))
def TrainTest():
    pass
def AllTest():
    EncoderTest()
    DecoderTest()
    WNetTest()
    TrainTest()
    print('WNet Passed All Tests!')
AllTest()