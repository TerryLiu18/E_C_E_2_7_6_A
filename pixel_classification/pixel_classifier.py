'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import os
import sys
import pdb
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH)
from logistic import Logistic

w_path = os.path.join(PATH, 'w.npy')
b_path = os.path.join(PATH, 'b.npy')

class PixelClassifier():
    def __init__(self, feature_size=3, class_num=3, iterations=1000):
        """
	    Initilize your classifier with any parameters and attributes you need
        """

        self.model = Logistic(iterations, feature_size, class_num)
        if os.path.exists(w_path) and os.path.exists(b_path):
            self.model.load_param(w_path, b_path)
        else:
            train_folder = os.path.join(PATH, 'data/training/')
            colors = ("blue", "green", "red")
            train_X , labels = [], [] 
            for i, folder in enumerate(train_folder + color for color in colors):
                X = read_pixels(folder)
                train_X.append(X)
                labels.append(np.ones(X.shape[0]) * i)
            train_X, labels = np.concatenate(train_X), np.concatenate(labels)
            self.model.fit(train_X, labels)
            self.model.save_param(w_path, b_path)

    def classify(self, X):
        """
         Classify a set of pixels into red, green, or blue
         :param X: n x 3 matrix of RGB values
         :return: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
        """
        ################################################################
        result = self.model.predict(X)
        return result

    def calc_accuracy(self, X, y):
        return np.mean(self.classify(X) == y)

if __name__ == "__main__":
    from generate_rgb_data import read_pixels

    folder = 'data/training'
    X1 = read_pixels(folder + '/red', verbose=True)
    X2 = read_pixels(folder + '/green')
    X3 = read_pixels(folder + '/blue')
    y1, y2, y3 = np.full(X1.shape[0], 1), np.full(X2.shape[0], 2), np.full(X3.shape[0], 3)
    X, y = np.concatenate((X1, X2, X3)), np.concatenate((y1, y2, y3))   

    print(X.shape)
    print(y.shape)
    pdb.set_trace()

    mymodel = PixelClassifier()
    # mymodel.fit(X, y)
    acc_0 = mymodel.calc_accuracy(X, y)
    print(acc_0)
    mymodel.load_param()
    acc = mymodel.calc_accuracy(X, y)
    print(acc)
