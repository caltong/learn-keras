#coding=utf-8
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# create data
X = np.linspace(-1,1,200)
np.random.shuffle(X)  # 随机数据
Make_Deviation = np.random.normal(0,0.05,200) # 加入偏差
Y = 0.5*X + 2 + Make_Deviation
#plot data


X_train, Y_train = X[:160], Y[:160] # 前160个作为train
X_test, Y_test = X[160:], Y[160:] # 后40个作为test

# 建立模型
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

# 损失函数 优化器
model.compile(loss='mse',optimizer='sgd')

# 训练
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost:',cost)

# test
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:',cost)
W, b = model.layers[0].get_weights()
print('Weight=',W,'\nbiases=',b)


# 画图
x = np.linspace(-1,1)
W = W[0]
y = W*x + b
plt.plot(x,y,color = 'red',linewidth = 5)
plt.scatter(X, Y)
plt.show()





