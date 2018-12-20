from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

(x_train, y_train),(x_test,y_test) = mnist.load_data()


print(x_train.shape)
print(x_train.shape[0])
x_train = x_train.reshape(x_train.shape[0],-1)/255.0
x_test = x_test.reshape(x_test.shape[0],-1)/255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)
print(x_train.shape)

model = Sequential()
model.add(Dense(32,input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])

print('Training ----------')
model.fit(x_train,y_train,epochs=2,batch_size=32)
print('\nTesting ----------')
loss, accuracy = model.evaluate(x_test,y_test)
print('test loss:',loss)
print('test accuracy:',accuracy)


