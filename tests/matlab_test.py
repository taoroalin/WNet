from scipy.io import loadmat
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load a matlab file
x = loadmat('../data/groundTruth/train/8049.mat')

# Get the amount of classes, amount of pixel rows and amount of pixel columns
amountClasses = len(x['groundTruth'][0])
pix_rows = len(x['groundTruth'][0][0]['Segmentation'][0][0])
pix_columns = len(x['groundTruth'][0][0]['Segmentation'][0][0][0])

print("Amount of classes" + str(amountClasses))
print("Amount of pixel rows" + str(pix_rows))
print("Amount of pixel columns" + str(pix_columns))

# for every class plot the segmentation
for i in range(amountClasses): 
    image_values = x['groundTruth'][0][i]['Boundaries'][0][0]
    class_map = np.zeros_like(image_values)
    class_map[np.where(image_values==np.max(image_values))] = 1

    # Plot the image with different colors for distinct values 
    ax = sns.heatmap(image_values, cmap="bi")
    # Plot the pixels with max values
    # ax = sns.heatmap(class_map)

    #Show image
    plt.show()


# Loop over every pixel
# for i in range(pix_rows):
#     for j in range(pix_columns):       
#         print(x['groundTruth'][0][0]['Segmentation'][0][0][i][j])
#         break




