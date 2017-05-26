import math
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))

def logProd(x):

    log_product = 0

    log_product = sum(x)

    return log_product

# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters alpha and beta, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
## Inputs ##
# XTrain - (n by V) numpy ndarray
# yTrain - 1D numpy ndarray of length n
# alpha - float
# beta - float

## Outputs ##
# D - (2 by V) numpy ndarray
def NB_XGivenY(XTrain, yTrain, alpha, beta):

    D = np.zeros([2, XTrain.shape[1]])
    Ny = [0,0]
    Ny[0] = len([y for y in yTrain if y == 0])
    Ny[1] = len([y for y in yTrain if y == 1])


    for r in range(len(D)):
        for c in range(len(D[0])):
            Nx_c = len([XTrain[i][c] for i in range(len(XTrain)) if XTrain[i][c] == 1 and yTrain[i] == r])
            D[r][c] = ( Nx_c + alpha - 1) / float((Ny[r] + alpha - 1 + beta - 1))

    return D
	
# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
	## Inputs ## 
	# yTrain - 1D numpy ndarray of length n

	## Outputs ##
	# p - float

	p =  sum([1 for i in range(len(yTrain)) if yTrain[i] == 0]) / float(len(yTrain))

	return p

# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
## Inputs ##
# D - (2 by V) numpy ndarray
# p - float
# XTest - (m by V) numpy ndarray

## Outputs ##
# yHat - 1D numpy ndarray of length m


def NB_Classify(D, p, XTest):

    yHat = np.ones(XTest.shape[0])

    for r in range(len(XTest)):
        val0 = []
        val1 = []
        for c in range(len(XTest[0])):
            if XTest[r][c] == 0:
                val0.append( 1 - D[0][c])
                val1.append(1 - D[1][c])
            else:
                val0.append(D[0][c])
                val1.append(D[1][c])

        prob0 = np.log(p) + logProd(np.log(val0))
        prob1 = np.log(1 - p) + logProd(np.log(val1))

        if prob0 > prob1:
            yHat[r] = 0

    return yHat

# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
	## Inputs ## 
	# yHat - 1D numpy ndarray of length m
	# yTruth - 1D numpy ndarray of length m
	
	## Outputs ##
	# error - float
    error =  len([yHat[i] for i in range(len(yHat)) if yHat[i] != yTruth[i]])/ float(len(yHat))
    return error