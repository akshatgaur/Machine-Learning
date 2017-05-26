import os
import csv
import numpy as np
import kmeans
from sklearn.decomposition import PCA

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Load numeric data files into numpy arrays
# X = np.genfromtxt(os.path.join(data_dir, 'kmeans_test_data.csv'), delimiter=',')

#X = np.array([(1.0,1.0),(1.5,2.0),(3.0,4.0),(5.0,7.0),(3.5,5.0),(4.5,5.0),(3.5,4.5)])
C = np.array([(1.0,1.0),(5.0,7.0)])
# TODO: Test update_assignments function, defined in kmeans.py

# a = kmeans.update_assignments(X, C)
# C_prime = kmeans.update_centers(X, C, a)
# a = kmeans.update_assignments(X,C_prime)
# C_p = kmeans.update_centers(X,C_prime, a)
# a = kmeans.update_assignments(X,C_p)
# Cpp = kmeans.update_centers(X,C_p, a)
# print a
# print Cpp
#
# C, a = kmeans.lloyd_iteration(X,C)
# print a
# print C
# TODO: Test update_centers function, defined in kmeans.py

# TODO: Test lloyd_iteration function, defined in kmeans.py

# TODO: Test kmeans_obj function, defined in kmeans.py

obj = 0.0
# for k in range(1000):
#     (best_C, best_a, best_obj) = kmeans.kmeans_cluster(X, 9, 'kmeans++', 1)
#     obj += best_obj
# print obj
# TODO: Run experiments outlined in HW6 PDF

# For question 9 and 10
# from sklearn.decomposition import PCA
mnist_X = np.genfromtxt(os.path.join(data_dir, 'mnist_data.csv'), delimiter=',')
pca = PCA(n_components=5)
mnist_X_reduced = pca.fit_transform(mnist_X)
(best_C, best_a, best_obj) = kmeans.kmeans_cluster(mnist_X_reduced, 3, 'fixed', 1)
print best_obj

inv_transform_C = pca.inverse_transform(best_C)
print (inv_transform_C)