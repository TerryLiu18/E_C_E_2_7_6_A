'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import os
import sys
import numpy as np
import cv2
import random
import pdb
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH)
from logistic2 import Logistic

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

class BinDetector():
	def __init__(self):
		"""
		Initilize your bin detector with the attributes you need,
		e.g., parameters of your classifier
		"""
        
		self.color_space = cv2.COLOR_BGR2RGB
		self.model = Logistic()
		
		w_path = os.path.join(dir_path, 'ww.npy')
		b_path = os.path.join(dir_path, 'bb.npy')
		if os.path.exists(w_path) and os.path.exists(b_path):
			self.load_param()
		else:
			print('parameter not found! start training...')
			self.training()

	def training(self):
		"""
		Train your color classifier if model parameters not found.
		"""
		print('parameter not found! start training...')
		# folder_path = os.path.join(dir_path, 'training')

		file_path = os.path.join(dir_path, 'data/selected_img.npy')
		img = np.load(file_path)
		print('img loaded!', img[:3, :])
		training_set = img[:, :3]
		training_label = img[:, 3]
		print('training set shape:', training_set.shape, 'training label shape:', training_label.shape)
		
		self.model.fit(training_set, training_label)
		self.model.save_param()

		# selected_img = np.unique(img, axis=0)
		# print(selected_img.shape)
		# print(selected_img)
		# for filename in os.listdir(file_path):
		# 	layers = np.load(os.path.join(folder_path, filename))
		# 	img_name = os.path.splitext(filename)[0] + ".jpg"
		# 	img = cv2.imread(os.path.join(folder_path, img_name))
		# 	img = cv2.cvtColor(img, self.color_space)
		# 	img = img.reshape([-1, 3])
		# 	layers = np.reshape(layers, (layers.shape[0], -1))
		# 	for layer_id in range(layers.shape[0]):
		# 		mask = layers[layer_id, :]
		# 		if np.sum(mask) > PIXEL_PER_IMG[layer_id]:
		# 			pixel_index = np.nonzero(mask)[0]
		# 			pixel_index = np.random.choice(pixel_index, size=(PIXEL_PER_IMG[layer_id], ), replace=False)
		# 			pixel_list.append(img[:, pixel_index])
		# 			label_list.append(np.ones(len(pixel_index)) * layer_id)


	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		img = cv2.cvtColor(img, self.color_space)
		img = img.astype(np.float32)/255.0
		pixels = img.reshape([-1, 3])
		label = self.model.predict(pixels)
		mask_img = np.reshape(label == 0, (img.shape[0], img.shape[1]))
		smoothed_mask = np.zeros_like(mask_img)
		# print('this is the masked output we obtain')

		# Smoothing
		for i in range(1, img.shape[0]-1):
			for j in range(1, img.shape[1]-1):
				if np.sum(mask_img[(i-1):(i+2), (j-1):(j+2)]) > 4:
					smoothed_mask[i, j] = 1
		mask_img = smoothed_mask
		# plt.imshow(smoothed_mask)
		# plt.show()
		# pdb.set_trace()
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

		# Replace this with your own approach 
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		label_mask = label(img)
		regions = regionprops(label_mask)
		boxes = []
		total_area = img.shape[0] * img.shape[1]
		for r in regions:
			if 0.05*total_area < r.area < 0.6 * total_area:
				boxes.append([r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2]])

		# Replace this with your own approach 
		# x = np.sort(np.random.randint(img.shape[0], size=2)).tolist()
		# y = np.sort(np.random.randint(img.shape[1], size=2)).tolist()
		# boxes = [[x[0], y[0], x[1], y[1]]]
		# boxes = [[182, 101, 313, 295]]
		return boxes

	def load_param(self):
		w_path = os.path.join(dir_path, 'ww.npy')
		b_path = os.path.join(dir_path, 'bb.npy')
		self.model.w = np.load(w_path)
		self.model.b = np.load(b_path)
		# self.model.param = {'w': self.w, 'b': self.b}
		print('parameter loaded！')



if __name__ == "__main__":
	detector = BinDetector()
	detector.load_param()
	# folder = './data/validation'
	folder = './data/training'


	for filename in os.listdir(folder):
		if os.path.splitext(filename)[1] == ".jpg":
			img = cv2.imread(os.path.join(folder,filename))
			# get the image mask
			mask = detector.segment_image(img)
			bbox = detector.get_bounding_boxes(mask)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			# display the labeled region and the image mask
			fig, (ax1, ax2) = plt.subplots(1, 2)
			fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])
			from matplotlib import patches
			if len(bbox):
				bbox = bbox[0]
				bbox_start = bbox[0:2]
				bbox_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
				rect = patches.Rectangle(bbox_start, bbox_size[0], bbox_size[1], linewidth=1, edgecolor='r', facecolor='none')
				ax1.add_patch(rect)
			ax1.imshow(img)
			ax2.imshow(mask)
			plt.show(block=True)
