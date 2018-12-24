import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# 用pickle打开cifar10数据集
def unpickle(file):
    with open(file,'rb') as fo:
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
#将cifa10 每张图片3072转换成1024*3 并且输出灰度
def get_photo_gray(pixel):
    assert len(pixel) == 3072
    r = pixel[0:1024]
    g = pixel[1024:2048]
    b = pixel[2048:3072]
    gray = 0.299*r+0.587*g+0.114*b
    return gray

data = unpickle('cifar-10-batches-py\data_batch_1')  #打开指定位置的pickle
x_train = data[b'data']   #flat数据
y_train = data[b'labels']  #标签
#x_train_rgb = [[0 for i in range(10000) for j in range(1024)]]
x_train_rgb = np.zeros((10000, 1024), dtype=np.float32)
for i in range(10000):
    x_train_rgb[i] = get_photo_gray(x_train[i])  #转换成灰度


x_train_rgb = x_train_rgb.reshape(x_train_rgb.shape[0],-1)/255.0
y_train = np_utils.to_categorical(y_train,num_classes=10)

model = Sequential()
model.add(Dense(1024,input_dim=1024))  #全连接层
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10))                 #全连接层
model.add(Activation('softmax'))

rmsprop = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)   #RMSprop 优化

model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_rgb,y_train,epochs=12,batch_size=16)   # 结果acc=0.1 flat后的数据acc=0.19
