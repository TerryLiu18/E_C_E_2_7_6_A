'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
from generate_rgb_data import read_pixels
from pixel_classifier import PixelClassifier

if __name__ == '__main__':
    # test the classifier

    folder = 'data/validation/blue'

    X = read_pixels(folder)
    myPixelClassifier = PixelClassifier()
    myPixelClassifier.load_param()
    y = myPixelClassifier.classify(X)
    print('Precision of blue: %f' % (sum(y == 3) / y.shape[0]))

    # folder = 'data/validation/green'
    # X = read_pixels(folder)
    # myPixelClassifier = PixelClassifier()
    # myPixelClassifier.load_param()
    # y = myPixelClassifier.classify(X)
    # print('Precision of green: %f' % (sum(y == 2) / y.shape[0]))
    #
    # folder = 'data/validation/red'
    # X = read_pixels(folder)
    # myPixelClassifier = PixelClassifier()
    # myPixelClassifier.load_param()
    # y = myPixelClassifier.classify(X)
    # print('Precision of red: %f' % (sum(y == 1) / y.shape[0]))