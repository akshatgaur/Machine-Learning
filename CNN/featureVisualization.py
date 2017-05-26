import numpy as np
import cnn_lenet
import testLeNet
import pickle
import matplotlib.pyplot as plt
# from skimage.io import imshow

def conv_net(params, layers, data, labels):
  """

  Args:
    params: a dictionary that stores hyper parameters
    layers: a dictionary that defines LeNet
    data: input data with shape (784, batch size)
    labels: label with shape (batch size,)

  Returns:
    cp: train accuracy for the train data
    param_grad: gradients of all the layers whose parameters are stored in params

  """
  l = len(layers)
  batch_size = layers[1]['batch_size']
  assert layers[1]['type'] == 'DATA', 'first layer must be data layer'

  output = {}
  output[1] = {}
  output[1]['data'] = data
  output[1]['height'] = layers[1]['height']
  output[1]['width'] = layers[1]['width']
  output[1]['channel'] = layers[1]['channel']
  output[1]['batch_size'] = layers[1]['batch_size']
  output[1]['diff'] = 0

  for i in range(2, 4):
    if layers[i]['type'] == 'CONV':
      output[i] = cnn_lenet.conv_layer_forward(output[i-1], layers[i], params[i-1])
    elif layers[i]['type'] == 'POOLING':
      output[i] = cnn_lenet.pooling_layer_forward(output[i-1], layers[i])
    elif layers[i]['type'] == 'IP':
      output[i] = cnn_lenet.inner_product_forward(output[i-1], layers[i], params[i-1])
    elif layers[i]['type'] == 'RELU':
      output[i] = cnn_lenet.relu_forward(output[i-1], layers[i])

  return output

# load the trained parameters
layers = testLeNet.get_lenet()
pickle_path = 'lenet.mat'
pickle_file = open(pickle_path, 'rb')
params = pickle.load(pickle_file)
pickle_file.close()

# get the forward result of cnn
_, _, _, _, xtest, ytest = cnn_lenet.load_mnist(False)
xtest = xtest[:, 0 : 2]
ytest = ytest[0]
layers[1]['batch_size'] = 2
output = conv_net(params, layers, xtest, ytest)

# get the first 3 layers output
layer1 = output[1]['data'][:, 0].reshape(28, 28)
layer2 = output[2]['data'][:, 0].reshape(24, 24, 20)
layer3 = output[3]['data'][:, 0].reshape(20, 12, 12)

tmp2 = layer2[:, :, 0]
tmp3 = layer3[:, :, 0]

# Visualize the first layer
plt.imshow(layer1, cmap=plt.get_cmap('gray'),vmin=0,vmax=1)
plt.title('Layer1: Input Image'), plt.xticks([]), plt.yticks([])
plt.savefig('Layer1 Input Image')
plt.show()

# Visualize the second layer
fig = plt.figure()
plt.title('Layer2: Convolution')
plt.axis('off')
for i in range(1, 21):
    a = fig.add_subplot(5, 4, i)
    # plt.delaxes()
    plt.imshow(layer2[:, :, i - 1], cmap=plt.get_cmap('gray'),vmin=layer2[:, :, i - 1].min(),vmax=layer2[:, :, i - 1].max())
    # plt.imshow(cmap=plt.get_cmap('gray'))
    plt.axis('off')
plt.savefig('Layer2 Convolution')
plt.show()


# Visualize the third layer
ig = plt.figure()
plt.title('Layer3: Maxpooling')
plt.axis('off')
for i in range(1, 21):
    a = ig.add_subplot(5, 4, i)
    plt.imshow(layer3[i - 1, :, :], cmap=plt.get_cmap('gray'),vmin=layer3[i - 1, :, :].min(),vmax=layer3[i - 1, :, :].max())
    plt.axis('off')
plt.savefig('Layer3 Maxpooling')
plt.show()
print "done"