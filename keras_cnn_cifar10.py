import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Convolution2D,Activation,MaxPooling2D,Flatten
from keras.optimizers import Adam

def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_photo_gray(pixel):
    assert len(pixel) == 3072
    r = pixel[0:1024]
    g = pixel[1024:2048]
    b = pixel[2048:3072]
    gray = 0.299*r+0.587*g+0.114*b
    return gray

data = unpickle('cifar-10-batches-py\data_batch_1')
x_train = data[b'data']
y_train = data[b'labels']

x_train_rgb = np.zeros((10000, 1024), dtype=np.float32)
for i in range(10000):
    x_train_rgb[i] = get_photo_gray(x_train[i])

x_train = x_train.reshape(x_train.shape[0],-1)/255.0
y_train = np_utils.to_categorical(y_train,num_classes=10)

