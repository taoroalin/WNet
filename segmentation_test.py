import torch
import seaborn as sns
import matplotlib.pyplot as plt

segment1 = torch.load('segment1.pt')
segment2 = torch.load('segment2.pt')
segment3 = torch.load('segment3.pt')
segment4 = torch.load('segment4.pt')
segment = segment2

sns.heatmap(segment, cmap="binary")
plt.show()
segment = torch.squeeze(segment)
segment_normalize = torch.round(torch.sigmoid(segment))
# segment_normalize = torch.nn.functional.softmax(segment3).data
sns.heatmap(segment, cmap="binary")
plt.show()