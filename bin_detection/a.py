import os
import numpy as np

path = 'data/labeled_data.npy'
path2 = 'data/selected_img2.npy'


img = np.load(path)
print(img.shape)


# img = img[:, :3]

# print(img[:10, :])
# count the number of rows whose last column is not zero
# normalize the first 3 column of each row
# img = img.astype(np.float32)
# img[:, :-1] = img[:, :-1] / 255.0
# np.save(path2, img)

# index = np.where(img[:, -1] == 0)
# print(len(index[0]))
# indx = np.argwhere(np.isnan(img))
# print(indx)


# # for the last column, replace 0 with 1, and 3 with 1


# label = img[:, 3]

# img[img[:,-1] == 3] = 1
# print(img[:3, :])
# print(img.shape)


# print(img.shape)
# print(img)

# selected_img = np.unique(img, axis=0)
# print(selected_img.shape)
# print(selected_img[300:303, :])
# np.save('./data/selected_img.npy', selected_img)

# b_path = 'b.npy'
# w_path = 'w.npy'

# b = np.load(b_path)
# print(b.shape)
# print(b)