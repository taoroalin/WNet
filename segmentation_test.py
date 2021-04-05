import torch
import seaborn as sns
import matplotlib.pyplot as plt
from utils.crf import dense_crf
from cv2 import imread, imwrite, resize

segment1 = torch.load('segment1.pt')
segment2 = torch.load('segment2.pt')
segment3 = torch.load('segment3.pt')
segment4 = torch.load('segment4.pt')
# segment = segment2

print(segment1)
segment = torch.stack([segment1, segment2, segment3, segment4])

# segment = torch.load('segment1.pt')

# sns.heatmap(segment, cmap="binary")
# plt.show()
# segment = torch.squeeze(segment)
# print(type(segment))
# segment = -torch.log(segment)
# segment_normalize = torch.round(torch.sigmoid(segment))
# segment_normalize = torch.nn.functional.softmax(segment3).data
orimg = imread("data2/images/train/8049.jpg")
img = resize(orimg,(224,224))
Q = dense_crf(img,segment.numpy())

print(Q)



sns.heatmap(Q[0], cmap="cubehelix")
plt.show()
sns.heatmap(Q[1], cmap="cubehelix")
plt.show()
sns.heatmap(Q[2], cmap="cubehelix")
plt.show()
sns.heatmap(Q[3], cmap="cubehelix")
plt.show()