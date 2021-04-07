import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3

# https://github.com/lucasb-eyer/pydensecrf

# def channelSplit(image):
#     return np.dsplit(image,image.shape[-1])

def dense_crf(img, output_probs):
    # Channels
    c = output_probs.shape[0]
    # Height
    h = output_probs.shape[1]
    # Width
    w = output_probs.shape[2]

    # Unary from softmax is the Uenc from paper (-log)
    # Check if we can input the stuff from the uenc
    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    print(img.shape)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    # What are these parameters --? see git
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q
