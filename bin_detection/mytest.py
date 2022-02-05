import os
import numpy as np
import matplotlib.pyplot as plt


filepath = './data/color_class/'
for f in os.listdir(filepath):
    if f.endswith('.npy'):
        img = np.load(filepath + f)
        
        print(img.shape)
        # imshow 2 images next to each other
        subplot = plt.subplot(1, 2, 1)
        subplot.imshow(img[0, :, :])
        a = img[0, :, :]
        # find where a is not zero
        index = np.where(a != 0)
        print("index: {}".format(index))
        b = a[index]
        print("b.shape: {}".format(b.shape))


        subplot = plt.subplot(1, 2, 2)
        subplot.imshow(img[1, :, :])
        print(img[0, :, :].shape)
        print(img[1, :, :].shape)
        # plt.imshow(img[0, :, :])
        # plt.show()
        # plt.imshow(img[1, :, :])
        plt.show()
