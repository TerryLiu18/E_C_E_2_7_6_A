import os, cv2
from roipoly import RoiPoly
from typing import List
import numpy as np
from matplotlib import pyplot as plt
import pdb



def manual_label(color_class, path):
    """
    
    Manually label training pairs.

    :param color_class: a tuple of color class
    :param path: saving path
    :return: None
    
    """
    data_folder = "./data/training"
    n_layers = len(color_class)

    all_output = []
    for filename in os.listdir(data_folder)[:20]:
        print("fig: {}".format(filename))
        img = cv2.imread(os.path.join(data_folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("Image shape: {}".format(img.shape))
        # pdb.set_trace()

        mask = np.zeros([n_layers]+ list(img.shape[0:2]))
        for i, color in enumerate(color_class):
            fig, ax = plt.subplots()
            print("Currently marking {}".format(color))
            ax.imshow(img)
            my_roi = RoiPoly(fig=fig, ax=ax, color='r')
            mask[i, :, :] = my_roi.get_mask(img)
            # get where mask is not zero
            index = np.where(mask[i, :, :] != 0)
            print("index: {}".format(index))
            # get the value of mask
            output = img[index]
            label = COLOR_CLASS[color]
            # insert color class into the last dimension
            output = np.insert(output, output.shape[-1], label, axis=output.ndim-1)
            all_output.append(output)
            print("output.shape: {}".format(output.shape))
    
    # concatenate all output
    labeled_data = np.concatenate(all_output, axis=0)
    print("labeled_data.shape: {}".format(labeled_data.shape))
    save_file_name = path + "/" + "labeled_data.npy"
    np.save(save_file_name, labeled_data)
    print("labeled_data[:10]: {}".format(labeled_data[:10,:]))
    # labeled_data[:, :-1] = labeled_data[:, :-1].as / 255.0
    # save_file_name = path + "/" + "labeled_data_normalized.npy"
    # np.save(save_file_name, labeled_data)

if __name__ == "__main__":
    # label_picture(["Blue", "Green", "Red", "OtherBlue"], "./data/4classes")
    COLOR_CLASS = {'blue': 1, 'nonblue': 0}
    PATH = "./data"
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    manual_label(('blue', 'nonblue'), PATH)