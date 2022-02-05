'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
class Logistic():
    def __init__(self, feature_size=3, class_num=2, iterations=1000):
        """
	    Initilize your classifier with any parameters and attributes you need
        """
        self.w = np.random.randn(feature_size, class_num)
        self.b = np.zeros((1, class_num))
        self.class_num = class_num
        self.feature_size = feature_size
        self.iter = iterations
        self.param = None
        self.loss_history = []

    def fit(self, X, y, lr=0.01):
        for i in range(self.iter):
            # idx = np.random.choice(X.shape[0], batch_size)
            # X_batch, y_batch = X[idx], y[idx]  # X_batch: batch_size x feature_size; y_batch: batch_size x 1
            # Y_batch = self.one_hot(y_batch)
            # loss = self.cross_entropy(y_batch, self.pass_forward(X_batch))
            loss = self.cross_entropy(y, self.pass_forward(X))
            # loss = self.cross_entropy(y, self.pass_forward(X))
            # print('iteration i: ',i)
            if i % 10 == 0:
                print("loss: ", loss)
            self.loss_history.append(loss)
            Y = self.one_hot(y)
            error = Y - self.pass_forward(X)  # batch_size x 3
            self.w += lr * np.dot(X.T, error)
            self.b += lr * np.mean(error, axis=0)
        self.save_param()

    def pass_forward(self, X):
        # print(self.w.shape)
        # print(X.shape)
        # print(self.b.shape)
        output = np.dot(X, self.w) + self.b
        assert np.isnan(output).any() == False, 'output contains nan'
        result = self.softmax(output)
        assert np.isnan(result).any() == False, 'result contains nan'
        return result
        # print(output.shape)
        # print(result.shape)

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

    def one_hot(self, y):
        # y is a vector of labels batch_size x 1, convert it to a matrix batch_size x 2
        # where the kth element of the ith row is 1 if the kth label is the correct label for the ith training example
        y = y.astype(int)
        m = y.shape[0]
        Y = np.zeros((m, self.class_num))
        for i in range(m):
            # Y[i, y[i] - 1] = 1
            Y[i, y[i]] = 1
        return Y

    def predict(self, X):
        """
         Classify a set of pixels into red, green, or blue
         :param X: n x 3 matrix of RGB values
         :return: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
        """
        ################################################################
        # YOUR CODE AFTER THIS LINE
        # self.load_param()
        output = self.pass_forward(X)
        # result = 1 + np.argmax(output, axis=1)
        result = np.argmax(output, axis=1)
        return result

        # y = 1 + np.random.randint(3, size=X.shape[0])
        ################################################################
        # return y

    def calc_accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def load_param(self):
        self.w = np.load('w.npy')
        self.b = np.load('b.npy')
        self.param = {'w': self.w, 'b': self.b}
        print('parameter loaded！')

    def save_param(self):
        np.save('w.npy', self.w)
        np.save('b.npy', self.b)


    def cross_entropy(self, y, probs):
        # print(probs.shape)
        # indx = np.argwhere(np.isnan(probs))
        # print(indx)

        assert np.isnan(probs).any() == False, 'probs contains nan'
        # return -1 * np.mean(y * np.log(probs))
        y = self.one_hot(y)
        return np.mean(np.sum(-y * np.log(probs), axis=1))


if __name__ == "__main__":
    # from generate_rgb_data import read_pixels

    # folder = 'data/training'
    # X1 = read_pixels(folder + '/red', verbose=True)
    # X2 = read_pixels(folder + '/green')
    # X3 = read_pixels(folder + '/blue')
    # y1, y2, y3 = np.full(X1.shape[0], 1), np.full(X2.shape[0], 2), np.full(X3.shape[0], 3)
    # X, y = np.concatenate((X1, X2, X3)), np.concatenate((y1, y2, y3))
    import random
    path = 'data/selected_img.npy'
    img = np.load(path)
    X = img[:, :-1].astype(np.float32)/255.0
    y = img[:, -1]

    mymodel = Logistic(class_num=2)
    acc_0 = mymodel.calc_accuracy(X, y)
    print("first time acc: ", acc_0)
    mymodel.fit(X, y)

    mymodel.load_param()
    acc = mymodel.calc_accuracy(X, y)
    print("acc: ", acc)
    