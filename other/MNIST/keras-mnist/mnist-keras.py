import numpy as np
np.random.seed(1337)  
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# 下载数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data('./mnist.npz') # 下载不来，可以手动下载下来后缀npz的数据，下载地址：https://s3.amazonaws.com/img-datasets/mnist.npz

# 数据预处处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255. 
X_test = X_test.reshape(X_test.shape[0], -1) / 255.  
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 不使用model.add()，用以下方式也可以构建网络
model = Sequential([
    Dense(400, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# 定义优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy']) # metrics赋值为'accuracy'，会在训练过程中输出正确率

# 这次我们用fit()来训练网路
print('Training ------------')
model.fit(X_train, y_train, epochs=4, batch_size=32)

print('\nTesting ------------')
# 评价训练出的网络
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)