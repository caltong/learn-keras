import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard

# 用pickle打开cifar10数据集
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# def get_photo(pixel):
#     assert len(pixel) == 3072
#     r = pixel[0:1024]
#     r = np.reshape(r,[32,32,1])
#     g = pixel[1024:2048]
#     g = np.reshape(g,[32,32,1])
#     b = pixel[2048:3072]
#     b = np.reshape(b,[32,32,1])
#     photo = np.concatenate([r,g,b],-1)
#     return photo
# 将cifa10 每张图片3072转换成1024*3 并且输出灰度
def get_photo_gray(pixel):
    assert len(pixel) == 3072
    r = pixel[0:1024]
    g = pixel[1024:2048]
    b = pixel[2048:3072]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


x_train = []
y_train = []
for i in [1, 2, 3, 4, 5]:  # 遍历数据集的全部5各包
    path = 'cifar-10-batches-py\data_batch_' + str(i)  # 获取路径
    data = unpickle(path)  # unpickle 打开数据集
    for i in range(10000):
        x_train.append(data[b'data'][i])  # 把所有数据读入x_train和y_train
        y_train.append(data[b'labels'][i])
print(len(x_train), len(x_train[0]))
print(len(y_train))
# data = unpickle('cifar-10-batches-py\data_batch_1')  #打开指定位置的pickle
# x_train = data[b'data']   #flat数据
# y_train = data[b'labels']  #标签
# x_train_rgb = [[0 for i in range(10000) for j in range(1024)]]
x_train_rgb = np.zeros((50000, 1024), dtype=np.float32)
for i in range(50000):
    x_train_rgb[i] = get_photo_gray(x_train[i])  # 转换成灰度

x_train_rgb = x_train_rgb.reshape(x_train_rgb.shape[0], -1) / 255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)

model = Sequential()
model.add(Dense(1024, input_dim=1024))  # 全连接层
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10))  # 全连接层
model.add(Activation('softmax'))

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)  # RMSprop 优化

model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         batch_size=32,  # 用多大量的数据计算直方图
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=True,  # 是否可视化梯度直方图
                         write_images=True,  # 是否可视化参数
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

model.fit(x_train_rgb, y_train, epochs=12, batch_size=128, callbacks=[tbCallBack])  # 结果acc=0.1 flat后的数据acc=0.19
