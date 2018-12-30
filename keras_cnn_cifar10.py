import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten
from keras.callbacks import TensorBoard


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


def get_photo_rgb(pixel):
    assert len(pixel) == 3072
    r = pixel[0:1024]
    g = pixel[1024:2048]
    b = pixel[2048:3072]
    return r, g, b

x_train = []
y_train = []
for i in [1, 2, 3, 4, 5]:
    path = 'cifar-10-batches-py/data_batch_' + str(i)
    data = unpickle(path)
    for i in range(10000):
        x_train.append(data[b'data'][i])  #把所有数据读入x_train和y_train
        y_train.append(data[b'labels'][i])
print('train_data shape\n',len(x_train), len(y_train))
# 转换RBG成三个通道
x_train_rgb = np.zeros((50000, 3, 1024), dtype=np.float32)
for i in range(50000):
    x_train_rgb[i][0],x_train_rgb[i][1],x_train_rgb[i][2] = get_photo_rgb(x_train[i])
print(len(x_train_rgb[0]),len(x_train_rgb[0][0]))
# 所有数据/255.0 标签改成标准式
x_train_rgb = x_train_rgb/255.0
# x_train_rgb = x_train_rgb.reshape(x_train_rgb.shape[0], -1) / 255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
print(x_train_rgb.shape,x_train_rgb[0].shape)
# 转换一维rgb成二维rgb
x_train_rgb_two_dimension = np.zeros((50000, 3, 32, 32), dtype=np.float32)
for i in range(50000):
    x_train_rgb_two_dimension[i][0] = np.array(x_train_rgb[i][0]).reshape(32, 32)
    x_train_rgb_two_dimension[i][1] = np.array(x_train_rgb[i][1]).reshape(32, 32)
    x_train_rgb_two_dimension[i][2] = np.array(x_train_rgb[i][2]).reshape(32, 32)

# test_train
path = 'cifar-10-batches-py/test_batch'
data = unpickle(path)                   #打开数据集 获取数据
print('test_data shape\n',len(data[b'data']))

x_test = []
y_test = []
for i in range(10000):
    x_test.append(data[b'data'][i])     #获取data信息
    y_test.append(data[b'labels'][i])   #获取data标签
print(len(x_test),len(x_test[0]))

x_test_rgb = np.zeros((10000,3,1024),dtype=np.float32) #转换成3通道
for i in range(10000):
    x_test_rgb[i][0],x_test_rgb[i][1],x_test_rgb[i][2] = get_photo_rgb(x_test[i])  #转换成3通道
print(x_test_rgb.shape)

x_test_rgb = x_test_rgb/255.0           #0-255 int 转换成 0-1 float32
y_test = np_utils.to_categorical(y_test,num_classes=10)  # 标准化
x_test_rgb_two_dimension = np.zeros((10000,3,32,32), dtype=np.float32)  #转换成2维
for i in range(10000):
    x_test_rgb_two_dimension[i][0] = np.array(x_test_rgb[i][0]).reshape(32,32)
    x_test_rgb_two_dimension[i][1] = np.array(x_test_rgb[i][1]).reshape(32,32)
    x_test_rgb_two_dimension[i][2] = np.array(x_test_rgb[i][2]).reshape(32,32)
print(x_test_rgb_two_dimension.shape,y_test.shape)

# model
model = Sequential()
model.add(Conv2D(
    32,
    kernel_size=(3, 3),
    data_format='channels_first',
))
model.add(Activation('relu'))
model.add(Conv2D(
    64,
    (3, 3),
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

tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)

model.fit(x_train_rgb_two_dimension, y_train, epochs=3, batch_size=128, callbacks=[tbCallBack])
score = model.evaluate(x_test_rgb_two_dimension,y_test,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
