import numpy as np
import os

file = 'labeled_data.npy'
f = np.load(file)
print(f.shape)
print(f[:10, :])
selected_img = np.unique(f, axis=0)
print("selected_img.shape: ", selected_img.shape)
print("selected_img: ", selected_img)
np.save('selected_img.npy', selected_img)
selected_img = selected_img.astype(np.float32)
selected_img[:, :-1] = selected_img[:, :-1] / 255.0
print("selected_img.shape: ", selected_img.shape)
print("selected_img: ", selected_img[1000:1013, :])


# print(selected_img.shape)
# print(selected_img[:10])
# print(selected_img.shape)
# save_file_name = "labeled_data_normalized.npy"
# np.save(save_file_name, selected_img)

# print(selected_img.shape)
# np.save('selected_img.npy', selected_img)

# print(f.shape)
# print(f[:5, :])


