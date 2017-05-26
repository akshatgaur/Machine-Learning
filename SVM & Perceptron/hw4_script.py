import os
import csv
import numpy as np
import perceptron
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

# Visualize the image
idx = 0
# datapoint = XTrain[idx, 1:]
# plt.imshow(datapoint.reshape((28,28), order = 'F'), cmap='gray')
# plt.show()

# TODO: Test perceptron_predict function, defined in perceptron.py

# TODO: Test perceptron_train function, defined in perceptron.py

# TODO: Test RBF_kernel function, defined in perceptron.py
#print perceptron.RBF_kernel(XTest[0], XTest[1],sigma=10)
a0 = np.zeros((len(XTrain),))
sigma = 1000
# 0.01 - .49
#0.1 - .49
#1 - 0.075
#10 - 0.05833
#100 - 0.08833
#1000 - 0.51

a = perceptron.kernel_perceptron_train(a0, XTrain, yTrain, 2, sigma)

miscl = 0
for i in range(len(XTest)):
    y_hat = perceptron.kernel_perceptron_predict(a, XTrain, yTrain, XTest[i], sigma)
    if y_hat != yTest[i]:
        miscl += 1

print float(miscl)/len(XTest)

#w0 = np.zeros((len(XTrain[0]),))
#w = perceptron.perceptron_train(w0, XTrain, yTrain, 10)

# miscl = 0
# for i in range(len(XTest)):
#     y_hat = perceptron.perceptron_predict(w, XTest[i])
#     if y_hat != yTest[i]:
#         miscl += 1
#
# print float(miscl)/len(XTest)


# TODO: Test kernel_perceptron_predict function, defined in perceptron.py

# TODO: Test kernel_perceptron_train function, defined in perceptron.py

# TODO: Run experiments outlined in HW4 PDF