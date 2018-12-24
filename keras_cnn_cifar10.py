import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_photo_gray(pixel):
    assert len(pixel) == 3072
    r = pixel[0:1024]
    g = pixel[1024:2048]
    b = pixel[2048:3072]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray
x_train = np.zeros((60000,3072),dtype=np.float32)
for i in (1,2,3,4,5):
    path = 'cifar-10-batches-py\data_batch_'+str(i)
    data = unpickle(path)
    x_train = x_train+data[b'data']
    y_train = y_train+data[b'labels']
print(len(x_train),len(y_train))
# 转换RBG成灰度
x_train_rgb = np.zeros((10000, 1024), dtype=np.float32)
for i in range(10000):
    x_train_rgb[i] = get_photo_gray(x_train[i])

# 所有数据/255.0 标签改成标准式
x_train_rgb = x_train_rgb.reshape(x_train_rgb.shape[0], -1) / 255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)

# 转换一维灰度成二维灰度
x_train_rgb_two_dimension = np.zeros((10000, 1, 32, 32), dtype=np.float32)
for i in range(10000):
    x_train_rgb_two_dimension[i][0] = np.array(x_train_rgb[i]).reshape(32, 32)

model = Sequential()
model.add(Conv2D(
    32,
    kernel_size=(3,3),

    data_format='channels_first',
))
model.add(Activation('relu'))
model.add(Conv2D(
    64,
    (3,3),
    data_format='channels_first',
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    2,
    2,
    data_format='channels_first',
))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train_rgb_two_dimension, y_train, epochs=1, batch_size=128)
