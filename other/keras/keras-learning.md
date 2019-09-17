#  一、keras 安装和入门

安装 keras：

1. 可以使用 pip 安装

   ``` 
   pip install keras
   ```

2. 也可以 ancoda 安装：

   ``` 
   conda install keras
   ```

测试是否安装成功：

``` python
 import keras
```

查看 keras 版本：

``` python
import keras
print(keras.__version__)
```



先看完该文：[一些基本概念 - Keras中文文档](https://keras-cn.readthedocs.io/en/latest/for_beginners/concepts/)

再看：[Keras TensorFlow教程：如何从零开发一个复杂深度学习模型 - SegmentFault 思否](https://segmentfault.com/a/1190000012645225)

其他阅读：

- [【私人笔记】深度学习框架keras踩坑记](https://mp.weixin.qq.com/s/suBo64ozWDSu-rQv118IVA)  [荐]



keras 模块：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190807170559.png)

注，常看到代码也会出现：

``` python
from keras.layers.core import Dense
```

查了下官方文档：“常用层对应于core模块，core内部定义了一系列常用的网络层，包括全连接、激活层等”——from：<https://keras-cn.readthedocs.io/en/latest/layers/core_layer/>



### 快速上手-mnist手写字识别

``` python
#-*-coding:utf-8-*-
from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.layers import Dense,Reshape,Dropout,Flatten,Conv2D,MaxPool2D
from keras.utils import to_categorical

def loadData():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    return x_train,y_train,x_test,y_test

def shuffle():
    pass

def createModel():
    model = Sequential()

    #reshape image to tensorflow backend shape (rows,clos,channels)
    model.add(Reshape(input_shape=(28,28),target_shape=(28,28,1)))
    # layer1-conv 卷积核大小：(5,5)，激活函数:relu,卷积核个数：32，第一层一定要指定input_shape
    model.add(Conv2D(32,kernel_size=(5,5),input_shape=(28,28,1),activation='relu'))
    # 2*2的最大池化
    model.add(MaxPool2D(pool_size=(2,2)))

    #layer2-conv 卷积核大小：(5,5)，激活函数:relu,卷积核个数：64
    model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
    # 2*2的最大池化
    model.add(MaxPool2D(pool_size=(2,2)))

    #数据一维化
    model.add(Flatten())

    #layer3-dense，输出1024
    model.add(Dense(1024,activation='relu'))
    #layer4-drop，断开神经元比例50%
    model.add(Dropout(0.5))

    #output，10个类，激活函数：softmax
    model.add(Dense(10,activation='softmax'))
    return model

def train():
    model = createModel()
    #损失函数：交叉熵 ，优化函数：adam，评估标准：精度
    model.compile(loss=categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
    x_train, y_train, x_test, y_test = loadData()
    #数据归一化
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    #转换为独热编码[1,2],[[0.1,0.5,...],[0.5,0.8,...]]
    y_train = to_categorical(y_train,10)
    y_test = to_categorical(y_test,10)
    #epochs：数据轮10遍，每个batch64，verbose=1： 输出日志信息
    model.fit(x=x_train,y=y_train,batch_size=64,epochs=1,shuffle=True,verbose=1, validation_data=(x_test,y_test))
    score = model.evaluate(x=x_test,y=y_test,batch_size=128,verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__=="__main__":
    train()
```





---



### 第一个例子：回归模型

首先我们在Keras中定义一个单层全连接网络，进行线性回归模型的训练：

``` 
# Regressor example
# Code: https://github.com/keloli/KerasPractise/edit/master/Regressor.py

import numpy as np
np.random.seed(1337)  
from keras.models import Sequential 
from keras.layers import Dense
import matplotlib.pyplot as plt

# 创建数据集
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # 将数据集随机化
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, )) # 假设我们真实模型为：Y=0.5X+2
# 绘制数据集plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]     # 把前160个数据放到训练集
X_test, Y_test = X[160:], Y[160:]       # 把后40个点放到测试集

# 定义一个model，
model = Sequential () # Keras有两种类型的模型，序贯模型（Sequential）和函数式模型
                      # 比较常用的是Sequential，它是单输入单输出的
model.add(Dense(output_dim=1, input_dim=1)) # 通过add()方法一层层添加模型
                                            # Dense是全连接层，第一层需要定义输入，
                                            # 第二层无需指定输入，一般第二层把第一层的输出作为输入

# 定义完模型就需要训练了，不过训练之前我们需要指定一些训练参数
# 通过compile()方法选择损失函数和优化器
# 这里我们用均方误差作为损失函数，随机梯度下降作为优化方法
model.compile(loss='mse', optimizer='sgd')

# 开始训练
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train) # Keras有很多开始训练的函数，这里用train_on_batch（）
    if step % 100 == 0:
        print('train cost: ', cost)

# 测试训练好的模型
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()    # 查看训练出的网络参数
                                        # 由于我们网络只有一层，且每次训练的输入只有一个，输出只有一个
                                        # 因此第一层训练出Y=WX+B这个模型，其中W,b为训练出的参数
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
```

最终的测试 cost 为：0.00313670327887，可视化结果如下图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190322153034.png)

### 第二个例子：手写数字识别

MNIST 数据集可以说是在业内被搞过次数最多的数据集了，毕竟各个框架的“hello world”都用它。这里我们也简单说一下在 Keras 下如何训练这个数据集：

``` 
# _*_ coding: utf-8 _*_
# Classifier mnist

import numpy as np
np.random.seed(1337)  
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# 下载数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data() # 下载不来，可以手动下载下来后缀npz的数据，下载地址：https://s3.amazonaws.com/img-datasets/mnist.npz

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
```

简单训练后得到：test loss: 0.0970609934615，test accuracy: 0.9743

### 第三个例子：加经典网络的预训练模型（以VGG16为例）

(1) 当服务器不能联网时，需要把模型 `*.h5` 文件下载到用户目录下的 `~/.keras/model`，模型的预训练权重在载入模型时自动载入

(2) 通过以下代码加载VGG16：

``` 
# 使用VGG16模型
from keras.applications.vgg16 import VGG16
print('Start build VGG16 -------')

# 获取vgg16的卷积部分，如果要获取整个vgg16网络需要设置:include_top=True
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

# 创建自己的输入格式
# if K.image_data_format() == 'channels_first':
#   input_shape = (3, img_width, img_height)
# else:
#   input_shape = (img_width, img_height, 3)

input = Input(input_shape, name = 'image_input') # 注意，Keras有个层就是Input层

# 将vgg16模型原始输入转换成自己的输入
output_vgg16_conv = model_vgg16_conv(input)

# output_vgg16_conv是包含了vgg16的卷积层，下面我需要做二分类任务，所以需要添加自己的全连接层
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dense(1, activation='softmax', name='predictions')(x)

# 最终创建出自己的vgg16模型
my_model = Model(input=input, output=x)

# 下面的模型输出中，vgg16的层和参数不会显示出，但是这些参数在训练的时候会更改
print('\nThis is my vgg16 model for the task')
my_model.summary()
```

### 其他 Keras 使用细节

#### 指定占用的GPU以及多GPU并行

查看 GPU 使用情况语句（Linux）：

``` 
# 1秒钟刷新一次
watch -n 1 nvidia-smi
```

指定显卡：

``` 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

这里指定了使用编号为 2 的GPU，大家可以根据需要和实际情况来指定使用的 GPU。

GPU 并行：

``` 
from model import unet
G = 3 # 同时使用3个GPU
with tf.device("/cpu:0"):
        M = unet(input_rows, input_cols, 1)
model = keras.utils.training_utils.multi_gpu_model(M, gpus=G)
model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train,
              batch_size=batch_size*G, epochs=nb_epoch, verbose=0, shuffle=True,
              validation_data=(X_valid, y_valid))

model.save_weights('/path/to/save/model.h5')
```

#### 查看网络结构的命令

查看搭建的网络：`print (model.summary())`

效果如图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190322153515.png)

#### 控制台显示模型，以及本地保存网络结构图

（1）控制台显示：

``` python
model = Model(inputs=xx, outputs=xx)
model.summary
```

这样在控制台下可以模型的参数等情况。

（2）本地保存网络结构图

``` 
# 你还可以用plot_model()来讲网络保存为图片
plot_model(my_model, to_file='my_vgg16_model.png')
```

补充1：[用图片存储和显示keras模型的结构](<https://blog.csdn.net/u012313895/article/details/79075750>)

keras 的可视化：

> keras 本身提供了可视化的库文件，但是由于 keras 的升级，已经抛弃了原来的接口和库，所以名字已经发生了改变，用网上的教程是走不通的。这里主要有三个库：
>
> - pydot  
> - pydot_ng
> - graphviz
>
> 由于keras新版本不再使用旧的 pydot 接口，所以查看源代码后实际使用的是 pydot_ng，原来的 pydot 只在异常处理的时候配合 pydotplus 使用。另外这里的 graphviz 是不能用 pip 安装的，安装完成后是不会识别的，一定一定要用 `sudo apt-get install graphviz`  安装。然后在程序中使用：
>
> ``` python
> from keras.utils.vis_utils import plot_model
> plot_model(Mod, to_file='model1.png',show_shapes=True)
> ```
>
> 就可以在指定的路径下生成模型对应的图片，下面是我 resnet 第一个区块所对应的图片。
>
> **!!!修正：** 导入 `keras.utils.vis_utils`  是不行的，实际导入的为 `from keras.utils import plot_model`。
>
> ``` python
> from keras.utils import plot_model
> plot_model(model, to_file="model_segnet.png", show_shapes=True)
> ```

补充2：[可视化 Visualization - Keras 中文文档](<https://keras.io/zh/visualization/>)

> `plot_model` 有 4 个可选参数:
>
> - `show_shapes` (默认为 False) 控制是否在图中输出各层的尺寸。
> - `show_layer_names` (默认为 True) 控制是否在图中显示每一层的名字。
> - `expand_dim`（默认为 False）控制是否将嵌套模型扩展为图形中的聚类。
> - `dpi`（默认为 96）控制图像 dpi。
>
> 此外，你也可以直接取得 `pydot.Graph` 对象并自己渲染它。 例如，ipython notebook 中的可视化实例如下：
>
> ``` python
> from IPython.display import SVG
> from keras.utils.vis_utils import model_to_dot
> 
> SVG(model_to_dot(model).create(prog='dot', format='svg'))
> ```

**!!!注意：** 这里有坑，按照网上教程 [手把手教你在win10下安装keras的plot_model环境 - 知乎](<https://zhuanlan.zhihu.com/p/32007274>)，分别使用 pip 安装如下：

``` python
pip install pydot-ng 
pip install graphviz
pip install pydot_ng
```

然后下载和安装客户端【[graphviz的客户端地址](<https://graphviz.gitlab.io/_pages/Download/Download_windows.html>)】，然后配置 bin 目录到 PATH 中。但是实际运行中，我一直出现如下这样的报错：

``` xml
OSError: pydot failed to call GraphViz.Please install GraphViz (https://www.graphviz.org/) and ensure that its executables are in the $PATH. 
```

一直没解决。后来尝试使用 conda 方式安装 graphviz：`conda install graphviz`，竟然可以了。（沃靠，这么坑我的吗）


#### 训练集与测试集图像的处理

```
from keras.preprocessing.image import ImageDataGenerator
print('Lodaing data -----------')
train_datagen=ImageDataGenerator()
test_datagen=ImageDataGenerator()
```

> *来源：[【Keras】Keras入门指南 - 简书](https://www.jianshu.com/p/e9c1e68a615e)*

 

---



# 二、keras API 学习

## 1. concatenate()函数

keras 中 concatenate 源代码如下：

``` python
def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.
    # Arguments
        tensors: list of tensors to concatenate.
        axis: concatenation axis.
    # Returns
        A tensor.
    """
    if axis < 0:
        rank = ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0
 
    if py_all([is_sparse(x) for x in tensors]):
        return tf.sparse_concat(axis, tensors)
    else:
        return tf.concat([to_dense(x) for x in tensors], axis)
```

可以看出 keras 的 concatenate() 函数是披了外壳的 tf.concat()。不过用法没有 tf.concat() 那么复杂。对tf.concat() 解释可以看这篇博文《[tf.concat()详解](https://blog.csdn.net/leviopku/article/details/82380118) 》，如果只想了解 concatenate 的用法，可以不用移步。

axis=n 表示**从第n个维度进行拼接**，对于一个三维矩阵，axis 的取值可以为[-3, -2, -1, 0, 1, 2]。虽然 keras 用模除允许 axis 的取值可以在这个范围之外，但不建议那么用。

可以通过如下小段代码来理解：

``` python
import numpy as np
import cv2
import keras.backend as K
import tensorflow as tf

t1 = K.variable(np.array([
    [
        [1, 2],
        [2, 3]
     ], 
    [
        [4, 4],
        [5, 3]
    ]
]))
t2 = K.variable(np.array([
    [
        [7, 4],
        [8, 4]
    ], 
    [
        [2, 10],
        [15, 11]
    ]
]))
d0 = K.concatenate([t1, t2], axis=-2)
d1 = K.concatenate([t1, t2], axis=1)
d2 = K.concatenate([t1, t2], axis=-1)
d3 = K.concatenate([t1, t2], axis=2)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(d0))
    print(sess.run(d1))
    print(sess.run(d2))
    print(sess.run(d3))
```

axis=-2，意思是从倒数第 2 个维度进行拼接，对于三维矩阵而言，这就等同于 axis=1。

axis=-1，意思是从倒数第 1 个维度进行拼接，对于三维矩阵而言，这就等同于 axis=2。

输出如下：

``` xml
d0:
[[[  1.   2.]
  [  2.   3.]
  [  7.   4.]
  [  8.   4.]]
 
 [[  4.   4.]
  [  5.   3.]
  [  2.  10.]
  [ 15.  11.]]]
 
d1:
[[[  1.   2.]
  [  2.   3.]
  [  7.   4.]
  [  8.   4.]]
 
 [[  4.   4.]
  [  5.   3.]
  [  2.  10.]
  [ 15.  11.]]]
 
d2:
[[[  1.   2.   7.   4.]
  [  2.   3.   8.   4.]]
 
 [[  4.   4.   2.  10.]
  [  5.   3.  15.  11.]]]
 
d3:
[[[  1.   2.   7.   4.]
  [  2.   3.   8.   4.]]
 
 [[  4.   4.   2.  10.]
  [  5.   3.  15.  11.]]]
```

参考：[keras中的K.concatenate()详解](<https://blog.csdn.net/leviopku/article/details/82380710>)

## 2. 两种训练模型方式fit和fit_generator

Keras深度学习库包括三个独立的函数，可用于训练您自己的模型：

- `.fit`
- `.fit_generator`
- `.train_on_batch`

### Keras .fit函数

``` python
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#读取数据
x_train = np.load("D:\\machineTest\\testmulPE_win7\\data_sprase.npy")[()]
y_train = np.load("D:\\machineTest\\testmulPE_win7\\lable_sprase.npy")

# 获取分类类别总数
classes = len(np.unique(y_train))

#对label进行one-hot编码，必须的
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_train = onehot_encoder.fit_transform(integer_encoded)

#shuffle
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=0)


model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=784))
model.add(Dense(units=classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=128)
score = model.evaluate(X_test, y_test, batch_size=128)

# #fit参数详情
# keras.models.fit(
# self,
# x=None, #训练数据
# y=None, #训练数据label标签
# batch_size=None, #每经过多少个sample更新一次权重，defult 32
# epochs=1, #训练的轮数epochs
# verbose=1, #0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# callbacks=None,#list，list中的元素为keras.callbacks.Callback对象，在训练过程中会调用list中的回调函数
# validation_split=0., #浮点数0-1，将训练集中的一部分比例作为验证集，然后下面的验证集validation_data将不会起到作用
# validation_data=None, #验证集
# shuffle=True, #布尔值和字符串，如果为布尔值，表示是否在每一次epoch训练前随机打乱输入样本的顺序，如果为"batch"，为处理HDF5数据
# class_weight=None, #dict,分类问题的时候，有的类别可能需要额外关注，分错的时候给的惩罚会比较大，所以权重会调高，体现在损失函数上面
# sample_weight=None, #array,和输入样本对等长度,对输入的每个特征+个权值，如果是时序的数据，则采用(samples，sequence_length)的矩阵
# initial_epoch=0, #如果之前做了训练，则可以从指定的epoch开始训练
# steps_per_epoch=None, #将一个epoch分为多少个steps，也就是划分一个batch_size多大，比如steps_per_epoch=10，则就是将训练集分为10份，不能和batch_size共同使用
# validation_steps=None, #当steps_per_epoch被启用的时候才有用，验证集的batch_size
# **kwargs #用于和后端交互
# )
# 
# 返回的是一个History对象，可以通过History.history来查看训练过程，loss值等等
```

在这里您可以看到我们提供的训练数据（X_train）和训练标签（y_train）。

然后，我们指示Keras允许我们的模型训练50个epoch，同时batch size为128。

对.fit的调用在这里做出两个主要假设：

- 我们的整个训练集可以放入RAM
- 没有数据增强（即不需要Keras生成器），相反，我们的网络将在原始数据上训练。

原始数据本身将适合内存，我们无需将旧批量数据从RAM 中移出并将新批量数据移入RAM。

此外，我们不会使用数据增强动态操纵训练数据。



### Keras fit_generator函数

``` python
# 第二种,可以节省内存
'''
Created on 2018-4-11

fit_generate.txt，后面两列为lable,已经one-hot编码
1 2 0 1
2 3 1 0
1 3 0 1
1 4 0 1
2 4 1 0
2 5 1 0

'''
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

count =1    
def generate_arrays_from_file(path):
    global count
    while 1:
        datas = np.loadtxt(path,delimiter=' ',dtype="int")
        x = datas[:,:2]
        y = datas[:,2:]
        print("count:"+str(count))
        count = count+1
        yield (x,y)
x_valid = np.array([[1,2],[2,3]])
y_valid = np.array([[0,1],[1,0]])
model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=2))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit_generator(generate_arrays_from_file("D:\\fit_generate.txt"),steps_per_epoch=10, epochs=2,max_queue_size=1,validation_data=(x_valid, y_valid),workers=1)
# steps_per_epoch 每执行一次steps,就去执行一次生产函数generate_arrays_from_file
# max_queue_size 从生产函数中出来的数据时可以缓存在queue队列中
# 输出如下:
# Epoch 1/2
# count:1
# count:2
# 
#  1/10 [==>...........................] - ETA: 2s - loss: 0.7145 - acc: 0.3333count:3
# count:4
# count:5
# count:6
# count:7
# 
#  7/10 [====================>.........] - ETA: 0s - loss: 0.7001 - acc: 0.4286count:8
# count:9
# count:10
# count:11
# 
# 10/10 [==============================] - 0s 36ms/step - loss: 0.6960 - acc: 0.4500 - val_loss: 0.6794 - val_acc: 0.5000
# Epoch 2/2
# 
#  1/10 [==>...........................] - ETA: 0s - loss: 0.6829 - acc: 0.5000count:12
# count:13
# count:14
# count:15
# 
#  5/10 [==============>...............] - ETA: 0s - loss: 0.6800 - acc: 0.5000count:16
# count:17
# count:18
# count:19
# count:20
# 
# 10/10 [==============================] - 0s 11ms/step - loss: 0.6766 - acc: 0.5000 - val_loss: 0.6662 - val_acc: 0.5000
```

对于小型，简单化的数据集，使用 Keras 的 .fit 函数是完全可以接受的。

这些数据集通常不是很具有挑战性，不需要任何数据增强。

但是，真实世界的数据集很少这么简单：

- 真实世界的数据集通常太大而无法放入内存中

- 它们也往往具有挑战性，要求我们执行数据增强以避免过拟合并增加我们的模型的泛化能力

在这些情况下，我们需要利用 Keras 的`.fit_generator`函数：

``` python
# initialize the number of epochs and batch size
EPOCHS = 100
BS = 32

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)
```

我们首先初始化将要训练的网络的epoch和batch size。

然后我们初始化 aug，这是一个Keras ImageDataGenerator对象，用于图像的数据增强，随机平移，旋转，调整大小等。执行数据增强是正则化的一种形式，使我们的模型能够更好的被泛化。但是，应用数据增强意味着我们的训练数据不再是“静态的” ——数据不断变化。根据提供给ImageDataGenerator的参数随机调整每批新数据。因此，我们现在需要利用Keras的.fit_generator函数来训练我们的模型。顾名思义，`.fit_generator` 函数假定存在一个为其生成数据的基础函数。该函数本身是一个 Python 生成器。

Keras在使用 `.fit_generator` 训练模型时的过程：

- Keras调用提供给 `.fit_generator` 的生成器函数（在本例中为 aug.flow）

- 生成器函数为 `.fit_generator` 函数生成一批大小为BS的数据

- `.fit_generator` 函数接受批量数据，执行反向传播，并更新模型中的权重

- 重复该过程直到达到期望的 epoch 数量

您会注意到我们现在需要在调用 `.fit_generator` 时提供 steps_per_epoch 参数（`.fit` 方法没有这样的参数）。

### steps_per_epoch

（接上。。

**为什么我们需要steps_per_epoch？** 

请记住，Keras 数据生成器意味着无限循环，它永远不会返回或退出。

由于该函数旨在无限循环，因此 Keras 无法确定一个epoch何时开始的，并且新的epoch何时开始。

因此，我们将**训练数据的总数除以批量大小的结果作为steps_per_epoch的值**。**一旦 Keras 到达这一步，它就会知道这是一个新的epoch。** 怎么理解？来看一个例子：

``` python
#……
train_flow=train_pic_gen.flow_from_directory(train_dir,(128,128),batch_size=32,class_mode='binary')
#……
model.fit_generator(
    morph.train_flow,steps_per_epoch=100,epochs=50,verbose=1,validation_data=morph.test_flow,validation_steps=100,
    callbacks=[TensorBoard(log_dir='./logs/1')]
)
#……
```

- 执行fit_generator时，由train_flow 数据流返回32（train_flow的batch_size的参数）张经过随机变形的样本，作为一个batch训练模型，
- 重复这一过程100（fit_generator的steps_per_epoch参数）次，一个epoch结束。一个epoch所用样本：batch_size乘以steps_per_epoch。 
- 当epoch=50（fit_generator的epochs参数）时，模型训练结束。

此外，根据官方文档：

- fit_generator的steps_per_epoch的建议值为样本总量除以train_flow的batch_size。

- fit_generator的steps_per_epoch，如果未指定（None）,则fit_generator的steps_per_epoch等于train_flow的batch_size。——form：[keras：fit_generator的训练过程](<https://blog.csdn.net/nima1994/article/details/80627504>)

参考：

- [keras 两种训练模型方式fit和fit_generator(节省内存)](<https://blog.csdn.net/u011311291/article/details/79900060>)
- [如何使用Keras fit和fit_generator（动手教程）](<https://blog.csdn.net/learning_tortosie/article/details/85243310>)

###validation_step

验证生成器与训练生成器完全相同，您可以定义每个时期将使用的批次数。

- 训练生成器将产生 steps_per_epoch 批次.
- 当纪元结束时，验证生成器将产生 validation_steps 批次.

但验证数据与培训数据完全无关。

根据培训批次，没有必要单独验证批次(我甚至会说这样做是没有意义的，除非你有非常具体的意图)。此外，训练数据中的样本总数与测试数据中的样本总数无关。

拥有多批次的目的只是为了节省计算机的内存，因此您可以一次测试一个较小的包。您可能会发现批量大小适合您的记忆或预期的训练时间并使用该大小。

也就是说， Keras 为您提供了一个完全免费的方法，因此您可以根据需要确定培训和验证批次。

**理想情况下，您一次使用所有验证数据。如果您仅使用部分验证数据，您将获得每个批次的不同指标，可能会让您认为您的模型实际上没有变得更糟或更好，您只是测量了不同的验证集。**

这就是为什么他们建议 `validation_steps = uniqueValidationData / batchSize`。从理论上讲，理论上你应该在每个时代训练你的整个数据.

所以,从理论上讲,每个时代都会产生：

- steps_per_epoch = TotalTrainingSamples / TrainingBatchSize
- validation_steps = TotalvalidationSamples / ValidationBatchSize

基本上，两个变量是：每个时期会产生多少批次。

这确保了在每个时代：

- 您完全训练整个训练集 
- 您确切地验证了整个验证集

参考：[参数 – Keras Sequential fit_generator参数列表中validation_steps的含义](https://codeday.me/bug/20190410/856147.html)

### keras训练过程之我的理解

程序会根据 steps_per_peoch 来确定一个 epoch 会要多少 steps 进行训练 --> 待一个 epoch 训练结束，再开始在验证集上进行验证（我的理解：**是整个验证集上的的验证**。可以看到 console 每个 epoch 结束都会打印出验证集上的 val_loss 和 val_acc），重复进行，直到到达设定的 epochs 的训练轮数。

## 3. Keras中的BatchNormalization函数

（1）

``` python
# import BatchNormalization
from keras.layers.normalization import BatchNormalization

# instantiate model
model = Sequential()

# we can think of this chunk as the input layer
model.add(Dense(64, input_dim=14, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.5))

# we can think of this chunk as the hidden layer    
model.add(Dense(64, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.5))

...
```

显然，批量标准化在激活功能之后在实践中更好地工作 - Claudiu

嗨@Claudiu，你介意扩大这个FYI吗？它似乎与上面的答案直接相矛盾。 - Ben Ogorek

@benogorek：当然，基本上我完全基于结果 这里 在relu表现更好之后放置批量规范的地方。 FWIW我没有成功地在我试过的一个网上以这种方式应用它 - Claudiu

有趣。为了跟进，如果你继续阅读该摘要，它说他们最好的模型[GoogLeNet128_BN_lim0606]实际上在ReLU之前有BN层。因此，虽然激活后的BN可以提高孤立情况下的准确性，但是在构建整个模型时，在执行最佳之前。可能在激活后放置BN可能会提高准确性，但可能依赖于问题。 - Lucas Ramadan

哇，那个基准真的很有趣。对于那里到底发生了什么，有没有人有任何直觉？为什么偏移和扩展激活会更好 后 非线性？是因为测试对象和游戏必须处理较少的变化或类似的东西，因此当训练数据不充足时，模型会更好地概括吗？ - Carl Thomé

（2）

现在几乎成了一个趋势 `Conv2D` 接下来是 `ReLu` 接下来是 `BatchNormalization` 层。所以我编写了一个小函数来立即调用所有这些函数。使模型定义看起来更清晰，更易于阅读。

```python
def Conv2DReluBatchNorm(n_filter, w_filter, h_filter, inputs):
    return BatchNormalization()(Activation(activation='relu')(Convolution2D(n_filter, w_filter, h_filter, border_mode='same')(inputs)))
```

关于是否应该在当前层的非线性或前一层的激活之前应用BN，该线程存在一些争论。

虽然没有正确的答案，批量标准化的作者说 **它应该在当前层的非线性之前立即应用。**原因（引自原始论文） -

“我们在此之前立即添加BN变换 非线性，通过归一化x = Wu + b。我们可以有 也标准化了层输入u，但因为你很可能 另一个非线性的输出，其分布的形状 在训练和约束期间可能会发生变化 它的第一和第二时刻不会消除协变量 转移。相比之下，吴+ b更有可能拥有 对称的非稀疏分布，即“更高斯” （Hyv¨arinen＆Oja，2000）;正常化它很可能 产生稳定分布的激活。

（3）

Keras现在支持 `use_bias=False` 选项，所以我们可以通过编写来节省一些计算

```
model.add(Dense(64, use_bias=False))
model.add(BatchNormalization(axis=bn_axis))
model.add(Activation('tanh'))
```

要么

```
model.add(Convolution2D(64, 3, 3, use_bias=False))
model.add(BatchNormalization(axis=bn_axis))
model.add(Activation('relu'))
```

（4）

它是另一种类型的图层，因此您应该将其作为图层添加到模型的适当位置。

```
model.add(keras.layers.normalization.BatchNormalization())
```

在这里查看示例： <https://github.com/fchollet/keras/blob/master/examples/kaggle_otto_nn.py>

参考：[我在哪里调用Keras中的BatchNormalization函数？](<http://landcareweb.com/questions/3901/wo-zai-na-li-diao-yong-keraszhong-de-batchnormalizationhan-shu>)

## 4. UpSampling2D与Conv2DTranspose、以及反卷积

（1）

UpSampling2D可以看作是 Pooling 的反向操作，就是采用Nearest Neighbor interpolation来进行放大，说白了就是复制行和列的数据来扩充 feature map 的大小。反向梯度传播的时候，应该就是每个单元格的梯度的和（猜测）。

Conv2DTranspose 就是正常卷积的反向操作，无需多讲。——from：https://www.zhihu.com/question/290376931/answer/471494441

（2）

UpSampling2D：

``` python
#coding:utf-8
import numpy as np
####keras 采用最邻近插值算法进行upsampling
def UpSampling2D(input_array,strides=(2,2)):
    h,w,n_channels = input_array.shape
    new_h,new_w = h*strides[0],w*strides[1]
    output_array=np.zeros((new_h,new_w,n_channels),dtype='float32')
    for i in range(new_h):
        for j in range(new_w):
            y=int(i/float(strides[0]))
            x=int(j/float(strides[1]))
            output_array[i,j,:]=input_array[y,x,:]
    return output_array
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from Convolution1 import Convolution2D
# if padding == 'valid':
#     dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
# elif padding == 'full':
#     dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
# elif padding == 'same':
#     dim_size = dim_size * stride_size
###(stride-1)*(img.shape-1)+img.shape

input_data=[ [[1,0,1,1],
              [0,2,1,1],
              [1,1,0,1],
              [1, 1, 0, 1]],
             [[2,0,2,1],
              [0,1,0,1],
              [1,0,0,1],
              [1, 1, 0, 1]],
             [[1,1,1,1],
              [2,2,0,1],
              [1,1,1,1],
              [1, 1, 0, 1]],
             [[1,1,2,1],
              [1,0,1,1],
              [0,2,2,1],
              [1, 1, 0, 1]]]
weights_data=[ [[[ 1, 0, 1],
                 [-1, 1, 0],
                 [ 0,-1, 0]],
                [[-1, 0, 1],
                 [ 0, 0, 1],
                 [ 1, 1, 1]],
                [[ 0, 1, 1],
                 [ 2, 0, 1],
                 [ 1, 2, 1]
                 ],
                [[ 1, 1, 1],
                 [ 0, 2, 1],
                 [ 1, 0, 1]
                 ]],
               [[[ 1, 0, 2],
                 [-2, 1, 1],
                 [ 1,-1, 0]
                 ],
                [[-1, 0, 1],
                 [-1, 2, 1],
                 [ 1, 1, 1]
                 ],
                [[ 0, 0, 0],
                 [ 2, 2, 1],
                 [ 1,-1, 1]
                 ],
                [[ 2, 1, 1],
                 [ 0,-1, 1],
                 [ 1, 1, 1]
                 ]] ]

a = np.asarray(input_data)
b = np.asarray(weights_data)
print('input_data',a.shape)
print('weight_data',b.shape)
def ConvTranspose2D(input_array,kernels,strides = (2,2),padding = 'same'):
    s_h,s_w = strides
    h,w,n = input_array.shape
    filters_num,k_h,k_w,n_channels = kernels.shape
    new_h,new_w = (s_h-1)*(h+1)+h,(s_w-1)*(w+1)+w
    tmp_array = np.zeros(shape=(new_h,new_w,n))
    y_range = range(s_h-1,new_h,s_h)
    x_range = range(s_w-1,new_w,s_w)
    for j in y_range:
        for i in x_range:
            tmp_array[j,i,:] = input_array[j//s_h,i//s_w,:]
    if padding == 'same':
        padding_h = new_h-1+k_h-new_h
        padding_w = new_w-1+k_w-new_w
        top_padding = padding_h // 2
        bottom_padding = padding_h - top_padding
        left_padding = padding_w // 2
        right_padding = padding_w - left_padding
        # print(origin_matrix.shape)
        tmp_array = np.pad(tmp_array, ((top_padding, bottom_padding), (left_padding, right_padding), (0, 0)),
                      mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        print(tmp_array.shape)
        result = Convolution2D(tmp_array,kernels,kernel_b=None,stride=(1,1),padding= 'valid')
        print(result.shape)
        print(result[:h*2,:w*2,0])
```

Conv2DTranspose：

``` python
import tensorflow as tf

def tf_conv2d_transpose(input,weights):
    #input_shape=[n,height,width,channel]
    input_shape = input.get_shape().as_list()
    #weights shape=[height,width,out_c,in_c]
    weights_shape=weights.get_shape().as_list()
    output_shape=[input_shape[0], input_shape[1]*2 , input_shape[2]*2 , weights_shape[2]]
    print("output_shape:",output_shape)
    deconv=tf.nn.conv2d_transpose(input,weights,output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME')
    return deconv
def main(input_data,weights_data):
    weights_np=np.asarray(weights_data,np.float32) #将输入的每个卷积核旋转180°
    weights_np=np.rot90(weights_np,2,(2,3))
    const_input = tf.constant(input_data , tf.float32)
    const_weights = tf.constant(weights_np , tf.float32 )
    input = tf.Variable(const_input,name="input") #[c,h,w]------>[h,w,c]
    input=tf.transpose(input,perm=(1,2,0)) #[h,w,c]------>[n,h,w,c]
    input=tf.expand_dims(input,0) #weights shape=[out_c,in_c,h,w]
    weights = tf.Variable(const_weights,name="weights") #[out_c,in_c,h,w]------>[h,w,out_c,in_c]
    weights=tf.transpose(weights,perm=(2,3,0,1)) #执行tensorflow的反卷积
    deconv=tf_conv2d_transpose(input,weights)
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    deconv_val = sess.run(deconv)
    hwc=deconv_val
    hwc = np.squeeze(hwc,0)
    print(hwc[:,:,0])


if __name__ == '__main__':

    input_data1 = np.asarray(input_data,dtype='float32').transpose((1,2,0))
    weights_data1 = np.asarray(weights_data,dtype='float32').transpose((0,2,3,1))
    input_data = np.asarray(input_data, dtype='float32')
    weights_data = np.asarray(weights_data, dtype='float32')
    #print(input_data.shape)
    #print(weights_data.shape)

    ConvTranspose2D(input_data1,weights_data1)
    main(input_data,weights_data)
```

参考：[UpSampling2D、Conv2DTranspose - zh_JNU的博客 - CSDN博客](<https://blog.csdn.net/zh_JNU/article/details/80986786>)

关于 Conv2DTranspose 输出大小计算：[keras的Conv2DTranspose与Conv2D输出大小](<https://blog.csdn.net/nima1994/article/details/83959495>)

### 补充：keras 中 Conv2DTranspose层

``` python
keras.layers.convolutional.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

该层是转置的卷积操作（反卷积）。需要反卷积的情况通常发生在用户想要对一个普通卷积的结果做反方向的变换。例如，将具有该卷积层输出shape的tensor转换为具有该卷积层输入shape的tensor。同时保留与卷积层兼容的连接模式。

当使用该层作为第一层时，应提供`input_shape`参数。例如`input_shape = (3,128,128)`代表128*128的彩色RGB图像

参数：

``` xml
filters：卷积核的数目（即输出的维度）
kernel_size：单个整数或由两个个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
dilation_rate：单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
use_bias:布尔值，是否使用偏置项
kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
kernel_regularizer：施加在权重上的正则项，为Regularizer对象
bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
activity_regularizer：施加在输出上的正则项，为Regularizer对象
kernel_constraints：施加在权重上的约束项，为Constraints对象
bias_constraints：施加在偏置上的约束项，为Constraints对象
```

参考：<https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/>

### 补充，在 tensorflow 中：tf.layers.conv2d_transpose（和tf.nn.conv2d_transpose？）

（1）[tf.layers.conv2d_transpose 反卷积](<https://blog.csdn.net/weiwei9363/article/details/78954063>)

参数：

``` xml
conv2d_transpose( 
inputs, 
filters, 
kernel_size, 
strides=(1, 1), 
padding=’valid’, 
data_format=’channels_last’, 
activation=None, 
use_bias=True, 
kernel_initializer=None, 
bias_initializer=tf.zeros_initializer(), 
kernel_regularizer=None, 
bias_regularizer=None, 
activity_regularizer=None, 
kernel_constraint=None, 
bias_constraint=None, 
trainable=True, 
name=None, 
reuse=None 
)
```

比较关注的参数：

- inputs: 输入的张量
- filters: 输出卷积核的数量
- kernel_size : 在卷积操作中卷积核的大小
- strides: （不太理解，我直接理解成放大的倍数）
- padding : ‘valid’ 或者 ‘same’。

反卷积的过程：

- Step 1 扩充: 将 inputs 进行填充扩大。扩大的倍数与strides有关。扩大的方式是在元素之间插strides - 1 个 0

- Step 2 卷积: 对扩充变大的矩阵，用大小为kernel_size卷积核做卷积操作，这样的卷积核有filters个，并且这里的步长为1(与参数strides无关，一定是1)

举个例子：

- inputs：[ [1, 1], [2,2] ]
- strides = 2(扩大2倍)
- filters = 1
- kernel_size = 3(假设核的值都是1)
- padding = ‘same’

代码：

``` python
a = np.array([[1,1],[2,2]], dtype=np.float32)
# [[1,1],
#  [2,2]]

# tf.layers.conv2d_transpose 要求输入是4维的
a = np.reshape(a, [1,2,2,1])

# 定义输入
x = tf.constant(a,dtype=tf.float32)
# 进行tf.layers.conv2d_transpose
upsample_x = tf.layers.conv2d_transpose(x, 1, 3, strides=2, padding='same', kernel_initializer=tf.ones_initializer())
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(upsample_x))
    # [[[[1],[1],[2],[1]],
    #   [[1],[1],[2],[1]],
    #   [[3],[3],[6],[3]],
    #   [[2],[2],[4],[2]]]]
```

（2）[【TensorFlow】tf.nn.conv2d_transpose是怎样实现反卷积的？](<https://blog.csdn.net/mao_xiao_feng/article/details/71713358>)

今天来介绍一下Tensorflow里面的反卷积操作，网上反卷积的用法的介绍比较少，希望这篇教程可以帮助到各位

反卷积出自这篇论文：Deconvolutional Networks，有兴趣的同学自行了解

首先无论你如何理解反卷积，请时刻记住一点，反卷积操作是卷积的反向

如果你随时都记住上面强调的重点，那你基本就理解一大半了，接下来通过一些函数的介绍为大家强化这个观念

conv2d_transpose(value, filter, output_shape, strides, padding="SAME", data_format="NHWC", name=None)

``` xml
除去name参数用以指定该操作的name，与方法有关的一共六个参数：
第一个参数value：指需要做反卷积的输入图像，它要求是一个Tenso
第二个参数filter：卷积核，它要求是一个Tensor，具有[filter_height, filter_width, out_channels, in_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，卷积核个数，图像通道数]
第三个参数output_shape：反卷积操作输出的shape，细心的同学会发现卷积操作是没有这个参数的，那这个参数在这里有什么用呢？下面会解释这个问题
第四个参数strides：反卷积时在图像每一维的步长，这是一个一维的向量，长度4
第五个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
第六个参数data_format：string类型的量，'NHWC'和'NCHW'其中之一，这是tensorflow新版本中新加的参数，它说明了value参数的数据格式。'NHWC'指tensorflow标准的数据格式[batch, height, width, in_channels]，'NCHW'指Theano的数据格式,[batch, in_channels，height, width]，当然默认值是'NHWC'
 
开始之前务必了解卷积的过程，参考我的另一篇文章：http://blog.csdn.net/mao_xiao_feng/article/details/53444333
```

（剩下内容略。。。



## 5. keras 中实现简单的反卷积

我这里将反卷积分为两个操作，一个是 UpSampling2D()，**用上采样将原始图片扩大**，然后**用 Conv2D() 这个函数进行卷积操作**，**就可以完成简单的反卷积。** 

。。。

这是核心代码，也就是 UpSampling2D() 函数就是一个 K.resize_imagse 操作，我这里 backend 用的是tensorflow，关于这个函数的解释在这里[点击打开链接](https://tensorflow.google.cn/api_docs/python/tf/image/resize_images)

``` python
tf.image.resize_images(
    images,
    size,
    method=ResizeMethod.BILINEAR,
    align_corners=False
)
```

method can be one of:

- ResizeMethod.BILINEAR: Bilinear interpolation.
- ResizeMethod.NEAREST_NEIGHBOR: Nearest neighbor interpolation.
- ResizeMethod.BICUBIC: Bicubic interpolation.
- ResizeMethod.AREA: Area interpolation.

UpSampling2D(size=(2,2)) 就可以将图片扩大1倍，比如原来为28x28的图片，就会变为56x56,

接下来就可以进行卷积操作：

``` python
keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

参考：[keras中实现简单的反卷积](<https://blog.csdn.net/huangshaoyin/article/details/81004301>)

### 在 tensorflow 中： tf.image.resize_images

tensorflow 里面用于改变图像大小的函数是 `tf.image.resize_images(image, （w, h）, method)`：image 表示需要改变此存的图像，第二个参数改变之后图像的大小，method 用于表示改变图像过程用的差值方法。

0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法。

例如：

``` python
import matplotlib.pyplot as plt;
import tensorflow as tf;
 
image_raw_data_jpg = tf.gfile.FastGFile('11.jpg', 'r').read()
 
with tf.Session() as sess:
	img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
	img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
	resize_0 = tf.image.resize_images(img_data_jpg, (500, 500), method=0)
	resize_1 = tf.image.resize_images(img_data_jpg, (500, 500), method=1)
	resize_2 = tf.image.resize_images(img_data_jpg, (500, 500), method=2)
	resize_3 = tf.image.resize_images(img_data_jpg, (500, 500), method=3)
	
	print resize_0.get_shape
 
	plt.figure(0)
	plt.imshow(resize_0.eval())
	plt.figure(1)
	plt.imshow(resize_1.eval())
	plt.figure(2)
	plt.imshow(resize_2.eval())
	plt.figure(3)
	plt.imshow(resize_3.eval())
 
	plt.show()
```

参考：[tensorflow里面用于改变图像大小的函数](<https://blog.csdn.net/UESTC_C2_403/article/details/72699260>)

有个注意的地方，在训练模型使用如下代码情况下：

``` python
...
output   = Add(name='fc1_voc12')([conv38_1, conv38_2, conv38_3, conv38_4])
output   = Lambda(lambda image: tf.image.resize_images(image, (H,W)))(output)
...
```

本人 load_mode() 导入模型进行测试集上的测试时，会有如下报错：

``` python
 NameError: name 'tf' is not defined
```

可按照该文 [在tf.keras模型中使用tensorflow操作 ](https://codeday.me/bug/20190607/1194897.html)（网上还找到一篇关于该问题：[Keras model cannot be loaded if it contains a Lambda layer calling tf.image.resize_images · Issue #5298](<https://github.com/keras-team/keras/issues/5298>) 可以一起看看）方式解决：

> 这是因为在模型加载期间，tf 不会在重构 Lambda 图层的范围内导入。它可以通过向 load_model 提供custom_objects 字典来解决。
>
> ``` python
> m = tf.keras.models.load_model('1.h5', custom_objects={'tf': tf})
> ```

我按如上方式操作了，但还会继续报这样类似错误：

``` python
NameError: name 'H' is not defined
 
NameError: name 'W' is not defined
```

解决方式是，同样是添加 custom_objects 字典方式：

``` python
m = tf.keras.models.load_model('1.h5', custom_objects={'tf': tf, 'H':H, 'W':W})
```

> 注：记得前面按照训练模型一致定义 H、W。
>
> ``` python
> H = 256
> W = 256
> ```

另外，对于该操作，在 keras 代码中，还可以这样写：

``` python
from keras import backend as K

K.resize_images(x, x.shape[1]*2, x.shape[2]*2, "channels_last")
```

> 注：参考「使用抽象 Keras 后端编写新代码」 <https://keras.io/zh/backend/>

查下函数的定义，如下：

``` python
def resize_images(x,
                  height_factor,
                  width_factor,
                  data_format,
                  interpolation='nearest'):
```



## 6. SGD 随机梯度下降优化器参数设置

Keras 中包含了各式优化器供我们使用，但通常我会倾向于使用 SGD 验证模型能否快速收敛，然后调整不同的学习速率看看模型最后的性能，然后再尝试使用其他优化器。Keras 中文文档中对 SGD 的描述如下：

`keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)`：随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量

参数：

- lr：大或等于0的浮点数，学习率
- momentum：大或等于0的浮点数，动量参数
- decay：大或等于0的浮点数，每次更新后的学习率衰减值
- nesterov：布尔值，确定是否使用Nesterov动量

### Time-Based Learning Rate Schedule

Keras 已经内置了一个基于时间的学习速率调整表，并通过上述参数中的 `decay` 来实现，学习速率的调整公式如下：

``` python
LearningRate = LearningRate * 1/(1 + decay * epoch)
```

当我们初始化参数为：

``` python
LearningRate = 0.1
decay = 0.001
```

大致变化曲线如下（非实际曲线，仅示意）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429174447.png)

当然，方便起见，我们可以将优化器设置如下，使其学习速率随着训练轮次变化：

``` python
sgd = SGD(lr=learning_rate, decay=learning_rate/nb_epoch, momentum=0.9, nesterov=True)
```

### Drop-Based Learning Rate Schedule

另外一种学习速率的调整方法思路是保持一个恒定学习速率一段时间后立即降低，是一种突变的方式。通常整个变化趋势为指数形式。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429174516.png)

对应的学习速率变化公式如下：

``` python
LearningRate = InitialLearningRate * DropRate^floor(Epoch / EpochDrop)
```

实现需要使用 Keras 中的 `LearningRateScheduler` 模块：

``` python
from keras.callbacks import LearningRateScheduler
# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

lrate = LearningRateScheduler(step_decay)

# Compile model
sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss=..., optimizer=sgd, metrics=['accuracy'])

# Fit the model
model.fit(X, Y, ..., callbacks=[lrate])
```

参考：<https://blog.csdn.net/u012862372/article/details/80319166>

## 7. model.predict_classes(test) 和model.predict(test) 区别

在keras中做深度网络预测时，有这两个预测函数model.predict_classes(test) 和model.predict(test)，本例中是多分类，标签经过了one-hot编码，如[1,2,3,4,5]是标签类别，经编码后为[1 0 0 0 0],[0 1 0 0 0]...[0 0 0 0 1]

- model.predict_classes(test)预测的是类别，打印出来的值就是类别号。**同时只能用于序列模型来预测，不能用于函数式模型**

    ``` python
predict_test = model.predict_classes(X_test).astype('int')
  inverted = encoder.inverse_transform([predict_test])
  print(predict_test)
  print(inverted[0])
  ```
  
    ``` xml
[1 0 0 ... 1 0 0]
  [2. 1. 1. ... 2. 1. 1.]
    ```
  
- model.predict(test)预测的是数值,而且输出的还是5个编码值，不过是实数，预测后要经过argmax(predict_test,axis=1)

  ``` python
  predict_test = model.predict(X_test)
  predict = argmax(predict_test,axis=1)  #axis = 1是取行的最大值的索引，0是列的最大值的索引
  inverted = encoder.inverse_transform([predict])
  print(predict_test[0:3])
  print(argmax(predict_test,axis=1))
  print(inverted)
  ```

  ``` xml
  [[9.9992561e-01 6.9890179e-05 2.3676146e-06 1.9608513e-06 2.5582506e-07]
  [9.9975246e-01 2.3708738e-04 4.9365349e-06 5.2166861e-06 3.3735736e-07]
  [9.9942291e-01 5.5233808e-04 8.9857504e-06 1.5617061e-05 2.4388814e-07]]
  [0 0 0 ... 0 0 0]
  [[1. 1. 1. ... 1. 1. 1.]]
  ```

  由于前几个和后几个每个预测值编码都是第一列最大，所以索引是0，反编码后是1

参考：[model.predict_classes(test) 和model.predict(test) 区别](<https://blog.csdn.net/zds13257177985/article/details/80638384>)



# 三、keras中的loss、optimizer、metrics

用 keras 搭好模型架构之后的下一步，就是执行编译操作。在编译时，经常需要指定三个参数：

- loss
- optimizer
- metrics

这三个参数有两类选择：

- 使用字符串
- 使用标识符，如 keras.losses，keras.optimizers，metrics 包下面的函数

例如：

``` python
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
```

因为有时可以使用字符串，有时可以使用标识符，令人很想知道背后是如何操作的。下面分别针对 optimizer，loss，metrics 三种对象的获取进行研究。

**optimizer：**

一个模型只能有一个optimizer，在执行编译的时候只能指定一个optimizer。
在keras.optimizers.py中，有一个get函数，用于根据用户传进来的optimizer参数获取优化器的实例。

**loss：**

keras.losses函数也有一个get(identifier)方法。其中需要注意以下一点：

如果 identifier 是可调用的一个函数名，也就是一个自定义的损失函数，这个损失函数返回值是一个张量。这样就轻而易举的实现了自定义损失函数。除了使用 str 和 dict 类型的 identifier，我们也可以直接使用 keras.losses 包下面的损失函数。

**metrics：**

在 model.compile() 函数中，optimizer 和 loss 都是单数形式，只有 metrics 是复数形式。因为一个模型只能指明一个optimizer和loss，却可以指明多个metrics。metrics也是三者中处理逻辑最为复杂的一个。

在 keras 最核心的地方 `keras.engine.train.py` 中有如下处理 metrics 的函数。这个函数其实就做了两件事：

- 根据输入的 metric 找到具体的 metric 对应的函数
- 计算 metric 张量

在寻找 metric 对应函数时，有两种步骤：

- 使用字符串形式指明准确率和交叉熵
- 使用 `keras.metrics.py` 中的函数

参考：

- [keras中的loss、optimizer、metrics - weiyinfu - 博客园](<https://www.cnblogs.com/weiyinfu/p/9783776.html>)



# 四、keras自定义loss、评估函数







# 五、Keras在训练过程中如何计算accuracy？

（1）

accuracy就是仅仅是计算而不参与到优化过程，keras metric 就是每跑一个 epoch 就会打印给你结果。

自定义 acc 的写法：

``` python
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])

出自官方文件：
Metrics - Keras Documentation
```

（2）

metric 的作用本来就只是评价，不参与训练。如果你想要把这个东西添加到训练里面，可以重新设计 loss 函数，由原先的对比损失加上你的新东西。

参考：

- [Keras在训练过程中如何计算accuracy？ - 知乎](<https://www.zhihu.com/question/303791916>)



# 六、keras模型的加载和保存、预训练、fine-tuning等

可以使用`model.save(filepath)`将Keras模型和权重保存在一个HDF5文件中，该文件将包含：

- 模型的结构，以便重构该模型
- 模型的权重
- 训练配置（损失函数，优化器等）
- 优化器的状态，以便于从上次训练中断的地方开始

使用 `keras.models.load_model(filepath)` 来重新实例化你的模型，如果文件中存储了训练配置的话，该函数还会同时完成模型的编译。

``` python
from keras.models import load_model
 
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
 
# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```

如果你只是希望保存模型的结构，而不包含其权重或配置信息，可以使用：

``` python
# save as JSON
json_string = model.to_json()
 
# save as YAML
yaml_string = model.to_yaml()
```

当然，你也可以从保存好的json文件或yaml文件中载入模型：

``` python
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)
 
# model reconstruction from YAML
model = model_from_yaml(yaml_string)
```

如果需要保存模型的权重，可通过下面的代码利用HDF5进行保存。注意，在使用前需要确保你已安装了HDF5和其Python库h5py

``` python
model.save_weights('my_model_weights.h5')
```

如果你需要在代码中初始化一个完全相同的模型，请使用：

``` python
model.load_weights('my_model_weights.h5')
```

如果你需要加载权重到不同的网络结构（有些层一样）中，例如fine-tune或transfer-learning，你可以通过层名字来加载模型：

``` python
model.load_weights('my_model_weights.h5', by_name=True)
```

例如：

``` python
"""
假如原模型为：
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="dense_1"))
    model.add(Dense(3, name="dense_2"))
    ...
    model.save_weights(fname)
"""
# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name="dense_1"))  # will be loaded
model.add(Dense(10, name="new_dense"))  # will not be loaded
 
# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)
```

参考：[Keras模型的加载和保存、预训练、按层名匹配参数](<https://blog.csdn.net/dongapple/article/details/77530212>)  |  [关于Keras模型 - Keras中文文档](<https://keras-cn.readthedocs.io/en/latest/models/about_model/>)

## 应用 Applications - Keras 中文文档

Keras 的应用模块（keras.applications）提供了带有预训练权值的深度学习模型，这些模型可以用来进行预测、特征提取和微调（fine-tuning）。

当你初始化一个预训练模型时，会自动下载权重到 `~/.keras/models/` 目录下。

可用的模型——在 ImageNet 上预训练过的用于图像分类的模型：

- [Xception](https://keras.io/zh/applications/#xception)
- [VGG16](https://keras.io/zh/applications/#vgg16)
- [VGG19](https://keras.io/zh/applications/#vgg19)
- [ResNet50](https://keras.io/zh/applications/#resnet50)
- [InceptionV3](https://keras.io/zh/applications/#inceptionv3)
- [InceptionResNetV2](https://keras.io/zh/applications/#inceptionresnetv2)
- [MobileNet](https://keras.io/zh/applications/#mobilenet)
- [DenseNet](https://keras.io/zh/applications/#densenet)
- [NASNet](https://keras.io/zh/applications/#nasnet)
- [MobileNetV2](https://keras.io/zh/applications/#mobilenetv2)

所有的这些架构都兼容所有的后端 (TensorFlow, Theano 和 CNTK)，并且会在实例化时，根据 Keras 配置文件`〜/.keras/keras.json` 中设置的图像数据格式构建模型。举个例子，如果你设置 `image_data_format=channels_last`，则加载的模型将按照 TensorFlow 的维度顺序来构造，即「高度-宽度-深度」（Height-Width-Depth）的顺序。

**参考：[应用 Applications - Keras 中文文档](<https://keras.io/zh/applications/>)  [荐]**



# 七、keras中数据增强 ImageDataGenerator

（1）[图片数据集太少？看我七十二变，Keras Image Data Augmentation 各参数详解 - 知乎](<https://zhuanlan.zhihu.com/p/30197320>)

图像深度学习任务中，面对小数据集，我们往往需要利用Image Data Augmentation图像增广技术来扩充我们的数据集，而keras的内置ImageDataGenerator很好地帮我们实现图像增广。但是面对ImageDataGenerator中众多的参数，每个参数所得到的效果分别是怎样的呢？

我们先来看看ImageDataGenerator的官方说明：

``` python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=K.image_data_format())
```

（2）[面向小数据集构建图像分类模型 - Keras中文文档](<https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/image_classification_using_very_little_data/>)

为了尽量利用我们有限的训练数据，我们将通过一系列随机变换堆数据进行提升，这样我们的模型将看不到任何两张完全相同的图片，这有利于我们抑制过拟合，使得模型的泛化能力更好。

在Keras中，这个步骤可以通过`keras.preprocessing.image.ImageGenerator`来实现，这个类使你可以：

- 在训练过程中，设置要施行的随机变换
- 通过`.flow`或`.flow_from_directory(directory)`方法实例化一个针对图像batch的生成器，这些生成器可以被用作keras模型相关方法的输入，如`fit_generator`，`evaluate_generator`和`predict_generator`

现在让我们看个例子：

``` python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
```

上面显示的只是一部分选项，请阅读文档的相关部分来查看全部可用的选项。我们来快速的浏览一下这些选项的含义：

- `rotation_range`是一个0~180的度数，用来指定随机选择图片的角度。
- `width_shift`和`height_shift`用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比例。
- `rescale`值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
- `shear_range`是用来进行剪切变换的程度，参考[剪切变换](https://en.wikipedia.org/wiki/Shear_mapping)
- `zoom_range`用来进行随机的放大
- `horizontal_flip`随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
- `fill_mode`用来指定当需要进行像素填充，如旋转，水平和竖直位移时，如何填充新出现的像素

下面我们使用这个工具来生成图片，并将它们保存在一个临时文件夹中，这样我们可以感觉一下数据提升究竟做了什么事。为了使图片能够展示出来，这里没有使用`rescaling`

``` python
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
```

下面是一张图片被提升以后得到的多个结果：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190523111308.png)



# 个人学习笔记及遇到的坑记录

## model.fit_generator()

先看代码：

``` python
checkpoint = ModelCheckpoint(filepath=model_name + '_model.h5',
                             monitor='val_dice_coef',
                             save_best_only=True,
                             save_weights_only=True)
```



```python
history = model.fit_generator(data_generator('dataset_parser/data_remote_images.h5', TRAIN_BATCH, 'train'),
                              steps_per_epoch=7500 // TRAIN_BATCH,
                              validation_data=data_generator('dataset_parser/data_remote_images.h5', VAL_BATCH, 'val'),
                              validation_steps=2500 // VAL_BATCH,
                              callbacks=[checkpoint, train_check],
                              epochs=10,
                              verbose=1)
```

我们先看网上对这些函数参数的解读：

1、ModelCheckpoint

``` python
keras.callbacks.ModelCheckpoint(filepath,monitor='val_loss',verbose=0,save_best_only=False, save_weights_only=False, mode='auto', period=1) 
```

参数说明：

- filename：字符串，保存模型的路径
- monitor：需要监视的值
- verbose：信息展示模式，0或1(checkpoint的保存信息，类似Epoch 00001: saving model to ...)
- save_best_only：当设置为True时，监测值有改进时才会保存当前的模型（ the latest best model according to the quantity monitored will not be overwritten）
- mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
- save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
- period：CheckPoint之间的间隔的epoch数

——from：[Keras笔记——ModelCheckpoint](<https://blog.csdn.net/breeze5428/article/details/80875323>)

以下为另外一篇博文的摘录 <https://blog.csdn.net/googler_offer/article/details/80916002>：

``` xml
from keras.callbacks import Model
checkpoint = ModelCheckpoint(filepath=file_name(就是你准备存放最好模型的地方), 
monitor='val_acc'(或者换成你想监视的值,比如acc,loss,
val_loss,其他值应该也可以,还没有试),
verbose=1(如果你喜欢进度条,那就选1,如果喜欢清爽的就选0,verbose=冗余的),
save_best_only='True'(只保存最好的模型,也可以都保存),
mode='auto'(如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
period=1(checkpoints之间间隔的epoch数))
/*
filename：字符串，保存模型的路径
monitor：需要监视的值
verbose：信息展示模式，0或1
save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，
例如，当监测值为val_acc时，模式应为max，
当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
 
period：CheckPoint之间的间隔的epoch数
*/
```

**1）monitor**

以上可以看到 monitor 的默认的取值有 `acc`、`loss`、 `val_acc`、`val_loss` 表示什么意思呢？表示分别可以监测训练集准确率和 loss，验证集准确率和 loss，这个在控制台打印也可以看到显示的是如此：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190505110351.png)

但也有看到程序代码 monitor 取值为 `val_dice_coef`，这是为什么呢？这是因为该程序在性能评价的时候使用了自定义评价函数：

```python
def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
```

``` python
model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
```

所以 monitor 需要监测验证集准确率，这个情况下得使用 `val_dice_coef`。在控制台打印的时候可以看到显示的也是 `dice_coef`、`val_dice_coef` 字样：

``` xml
468/468 [==============================] - 445s 951ms/step - loss: 0.7548 - dice_coef: 0.6032 - val_loss: 1.3339 - val_dice_coef: 0.5238
```

否则显示的是 acc、val_acc 字样。若这个时候想要监测训练集准确率，那么 monitor 参数设为 `dice_coef` 即可。

注：**save_best_only 参数**，我的理解是，monitor 设置了监测指标，比如监测验证集 loss，那么 save_best_only=True 则表示保存最好的模型（即 验证集 loss 最低的时候）。

**2）steps_per_epoch**

见前文「steps_per_epoch」节。

> **训练数据的总数除以批量大小的结果作为steps_per_epoch的值**。**一旦 Keras 到达这一步，它就会知道这是一个新的epoch。**

**3）validation_steps**

**validation_steps**: 仅当 `validation_data` 是一个生成器时才可用。 每个 epoch 结束时验证集生成器产生的步数。它通常应该等于你的数据集的样本数量除以批量大小。可选参数 `Sequence`：如果未指定，将使用 `len(generator)` 作为步数。——from：<https://keras.io/zh/models/sequential/>





## EarlyStopping callbacks

1、EarlyStopping 是什么？

EarlyStopping是Callbacks的一种，callbacks用于指定在每个epoch开始和结束的时候进行哪种特定操作。

Callbacks中有一些设置好的接口，可以直接使用，如’acc’,’val_acc’,’loss’和’val_loss’等等。 

**EarlyStopping 则是用于提前停止训练的callbacks。**具体地，可以达到当训练集上的 loss 不在减小（即减小的程度小于某个阈值）的时候停止继续训练。

2、为什么要用EarlyStopping？

根本原因就是因为继续训练会导致测试集上的准确率下降。 

那继续训练导致测试准确率下降的原因猜测可能是1. 过拟合 2. 学习率过大导致不收敛 3. 使用正则项的时候，Loss的减少可能不是因为准确率增加导致的，而是因为权重大小的降低。

当然使用EarlyStopping也可以加快学习的速度，提高调参效率。

3、EarlyStopping的使用与技巧

一般是在model.fit函数中调用callbacks，fit函数中有一个参数为callbacks。注意这里需要输入的是list类型的数据，所以通常情况只用EarlyStopping的话也要是[EarlyStopping()]

EarlyStopping的参数有：

- monitor: 监控的数据接口，有’acc’,’val_acc’,’loss’,’val_loss’等等。正常情况下如果有验证集，就用’val_acc’或者’val_loss’。但是因为笔者用的是5折交叉验证，没有单设验证集，所以只能用’acc’了。
- min_delta：增大或减小的阈值，只有大于这个部分才算作improvement。这个值的大小取决于monitor，也反映了你的容忍程度。例如笔者的monitor是’acc’，同时其变化范围在70%-90%之间，所以对于小于0.01%的变化不关心。加上观察到训练过程中存在抖动的情况（即先下降后上升），所以适当增大容忍程度，最终设为0.003%。
- patience：能够容忍多少个epoch内都没有improvement。这个设置其实是在抖动和真正的准确率下降之间做tradeoff。如果patience设的大，那么最终得到的准确率要略低于模型可以达到的最高准确率。如果patience设的小，那么模型很可能在前期抖动，还在全图搜索的阶段就停止了，准确率一般很差。patience的大小和learning rate直接相关。在learning rate设定的情况下，前期先训练几次观察抖动的epoch number，比其稍大些设置patience。在learning rate变化的情况下，建议要略小于最大的抖动epoch number。笔者在引入EarlyStopping之前就已经得到可以接受的结果了，EarlyStopping算是锦上添花，所以patience设的比较高，设为抖动epoch number的最大值。
- mode: 就’auto’, ‘min’, ‘,max’三个可能。如果知道是要上升还是下降，建议设置一下。笔者的monitor是’acc’，所以mode=’max’。

min_delta和patience都和“避免模型停止在抖动过程中”有关系，所以调节的时候需要互相协调。通常情况下，min_delta降低，那么patience可以适当减少；min_delta增加，那么patience需要适当延长；反之亦然。

## Keras保存最好的模型并用tensorboard可视化

不管是Tensorboard还是保存最好的模型,都需要用到Keras的一个重要模块: **keras.callbacks**

比如Tensorboard是：

``` python
from keras.callback import Tensorboad
```

keras.callbacks在model.fit中发挥作用,写法是：

``` python
from keras.callbacks import Tensorboard
......
tensorboard = Tensorboard(log_dir='log(就是你想存事件的文件夹)')
callback_lists = [tensorboard]  #因为callback是list型,必须转化为list
model.fit(x_train,y_train,bach_size=batch_size,epochs=epoch,shuffle='True',verbose='True',callbacks=callback_lists)
```

效果如下：略。。。

似乎有点跑题了，也算是自己的一个复习，keras保存最好的模型也用到keras.callbacks，用法如下: ModelCheckpoint：

``` python
from keras.callbacks import ModelCheckpoint
```

``` python
checkpoint = ModelCheckpoint(filepath=file_name(就是你准备存放最好模型的地方), 
monitor='val_acc'(或者换成你想监视的值,比如acc,loss,
val_loss,其他值应该也可以,还没有试),
verbose=1(如果你喜欢进度条,那就选1,如果喜欢清爽的就选0,verbose=冗余的),
save_best_only='True'(只保存最好的模型,也可以都保存),
mode='auto'(如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
period=1(checkpoints之间间隔的epoch数))
```

``` python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

既然Tensorboard和ModelCheckpoint都属于keras.callbacks类,那我们把它们写到一起吧。

``` python
from keras.callbacks import Tensorboad
from keras.callbacks import ModelCheckpoint
.....
tensorboad = Tensorboard(log_dir='log')
checkpoint = ModelCheckpoint(filepath='file_name',monitor='val_acc',mode='auto' ,save_best_only='True')

callback_lists=[tensorboard,checkpoint]
......
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0.33,shuffle=1,verbose=1,callbacks=callback_lists)
```

这样即保存了模型，也可以在tensorboard中可视化。

再次使用模型的时候,需要调用：

``` python
from keras.models import save_model(load_mode)
......
model = load_model('best_weights.h5')
loss,acc = model.evaluate(x_test,y_test,batch_size=batch_size,verbose=0)

print('test_loss',loss,'test_accuracy',acc)
```

这个callbacks感觉很有用，ReduceLROnPlateau。

当评价指标不在提升时，减少学习率

当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。该回调函数检测指标的情况，如果在patience个epoch中看不到模型性能提升，则减少学习率(是不是很棒!)

``` xml
参数

monitor：被监测的量
factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
epsilon：阈值，用来确定是否进入检测值的“平原区”
cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
min_lr：学习率的下限
```

——from：[Keras保存最好的模型](<https://www.jianshu.com/p/0711f9e54dd2>)

## load_model() 导入模型报错

我在训练完 HRNet 模型后，然后导入训练好的模型进行预测 `load_mode("hrnet.h5")` 发现报了如下字样的错误：

``` xml
str object has no attribute decode
```

经过网上找答案，发现导入模型设置 compile=False 即可解决 `load_model("hrnet.h5", compile=False)`。

参考的博文来自：[Keras加载预训练模型](<https://www.wandouip.com/t5i44145/>)，以下为原文。

Keras中存储模型主要有两种方式：

1.只存储模型中的权重参数：

```python
#save
model.save_weights('my_model_weights.h5')
#load
model.load_weights('my_model_weights.h5')
```

2.存储整个模型包括模型图的结构以及权重参数：

```python
#save
model.save('my_model.h5') 
#load
model = load_model('my_model.h5')
```

以上两步在存储和恢复通过系统模块构建的模型时就已经足够了，但是load_model()函数如果涉及到自定义的模块的时候还需要在加载模型的时候进行额外的声明。

以下举例进行说明：

比如训练模型的时候用到了自定义的模块AttentionLayer，那么在加载模型的时候需要在custom_objects的参数中声明对应的字典项，否则将会报模块未定义的错误。

```python
model = load_model('./model1/GRUAttention( 0.8574).h5', custom_objects={'AttentionLayer': AttentionLayer})
```

在训练的过程中有时候也会用到自定义的损失函数，这时候如果你加载模型知识为了进行预测不再其基础上再进行训练，那么加载模型的时候就没有必要在custom_objects参数中声明对应的字典项，只需要将compile参数设为False即可：

```python
model = load_model('./model1/GRUAttention(0.8574).h5', compile=False})
```

如果此时你好需要在加载后的模型上继续进行训练，那么声明损失函数对应的字典项就是必须的：

```python
model = load_model('./model1/GRUAttention(0.8574).h5', compile=True, custom_objects={'focal_loss_fixed':focal_loss})
```



---

# 整理于网络：keras 的坑

Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行。

Keras 的开发重点是支持快速的实验。能够以最小的时间把你的想法转换为实验结果，是做好研究的关键。本人是keras的忠实粉丝，可能是因为它实在是太简单易用了，不用多少代码就可以将自己的想法完全实现，但是在使用的过程中还是遇到了不少坑，本文做了一个归纳，供大家参考。

Keras 兼容的 Python 版本: Python 2.7-3.6。

**详细教程请参阅Keras官方中文文档：http://keras-cn.readthedocs.io/en/latest/** 

**1、Keras输出的loss，val这些值如何保存到文本中去：**

Keras中的fit函数会返回一个History对象，它的History.history属性会把之前的那些值全保存在里面，如果有验证集的话，也包含了验证集的这些指标变化情况，具体写法：

``` python
hist=model.fit(train_set_x,train_set_y,batch_size=256,shuffle=True,nb_epoch=nb_epoch,validation_split=0.1)
with open('log_sgd_big_32.txt','w') as f:
    f.write(str(hist.history))
```

我觉得保存之前的loss，val这些值还是比较重要的，在之后的调参过程中有时候还是需要之前loss的结果作为参考的，特别是你自己添加了一些自己的loss的情况下，但是这样的写法会使整个文本的取名比较乱，所以其实可以考虑使用Aetros的插件，Aetros网址，这是一个基于Keras的一个管理工具，可以可视化你的网络结构，中间卷积结果的可视化，以及保存你以往跑的所有结果，还是很方便的，就是有些不稳定，有时候会崩。。。

**history对象包含两个重要属性：**

- epoch：训练的轮数
- history：它是一个字典，包含val_loss,val_acc,loss,acc四个key。

**2、关于训练集，验证集和测试集：**

其实一开始我也没搞清楚这个问题，拿着测试集当验证集用，其实验证集是从训练集中抽取出来用于调参的，而测试集是和训练集无交集的，用于测试所选参数用于该模型的效果的，这个还是不要弄错了。。。在Keras中，验证集的划分只要在fit函数里设置validation_split的值就好了，这个对应了取训练集中百分之几的数据出来当做验证集。但由于shuffle是在validation _split之后执行的，所以如果一开始训练集没有shuffle的话，有可能使验证集全是负样本。测试集的使用只要在evaluate函数里设置就好了。

print model.evaluate（test_set_x，test_set_y ,batch_size=256）

这里注意evaluate和fit函数的默认batch_size都是32，自己记得修改。

**总结：**

- 验证集是在fit的时候通过validation_split参数自己从训练集中划分出来的；
- 测试集需要专门的使用evaluate去进行评价。

**3、关于优化方法使用的问题之学习率调整**

开始总会纠结哪个优化方法好用，但是最好的办法就是试，无数次尝试后不难发现，Sgd的这种学习率非自适应的优化方法，调整学习率和初始化的方法会使它的结果有很大不同，但是由于收敛确实不快，总感觉不是很方便，我觉得之前一直使用Sgd的原因一方面是因为优化方法不多，其次是用Sgd都能有这么好的结果，说明你网络该有多好啊。其他的Adam，Adade，RMSprop结果都差不多，Nadam因为是adam的动量添加的版本，在收敛效果上会更出色。所以如果对结果不满意的话，就把这些方法换着来一遍吧。 

**（1）方法一**：通过LearningRateScheduler实现学习率调整

有很多初学者人会好奇怎么使sgd的学习率动态的变化，其实Keras里有个反馈函数叫LearningRateScheduler，具体使用如下：

``` python
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)
sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
model.fit(train_set_x, train_set_y, validation_split=0.1, nb_epoch=200, batch_size=256, callbacks=[lrate])
```

上面代码是使学习率指数下降，具体如下图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190416095822.png)

**（2）方式二：**最直接的调整学习率方式

当然也可以直接在sgd声明函数中修改参数来直接修改学习率，学习率变化如下图：

``` python
sgd = SGD(lr=learning_rate, decay=learning_rate/nb_epoch, momentum=0.9, nesterov=True)
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190416095848.png)

具体可以参考这篇文章Using Learning Rate Schedules for Deep Learning Models in Python with Keras

除此之外，还有一种学利率调整方式，即

**（3）方法三：**通过ReduceLROnPlateau调整学习率

``` python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```

当评价指标不在提升时，减少学习率。当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。该回调函数检测指标的情况，如果在patience个epoch中看不到模型性能提升，则减少学习率

参数：

``` xml
monitor：被监测的量
factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
epsilon：阈值，用来确定是否进入检测值的“平原区”
cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
min_lr：学习率的下限
```

代码示例如下：

``` python
from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

model.fit(train_x, train_y, batch_size=32, epochs=5, validation_split=0.1, callbacks=[reduce_lr])
```

**4、如何用 Keras 处理超过内存的数据集？**

你可以使用 model.train_on_batch(x，y) 和 model.test_on_batch(x，y) 进行批量训练与测试。请参阅 模型文档。或者，你可以编写一个生成批处理训练数据的生成器，然后使用 model.fit_generator(data_generator，steps_per_epoch，epochs) 方法。

**5、Batchnormalization层的放置问题：**

BN层是真的吊，简直神器，除了会使网络搭建的时间和每个epoch的时间延长一点之外，但是关于这个问题我看到了无数的说法，对于卷积和池化层的放法，又说放中间的，也有说池化层后面的，对于dropout层，有说放在它后面的，也有说放在它前面的，对于这个问题我的说法还是试！虽然麻烦。。。但是DL本来不就是一个偏工程性的学科吗。。。还有一点是需要注意的，就是BN层的参数问题，我一开始也没有注意到，仔细看BN层的参数：

``` python
keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')
```

mode：整数，指定规范化的模式，取0或1

- 0：按特征规范化，输入的各个特征图将独立被规范化。规范化的轴由参数axis指定。注意，如果输入是形如（samples，channels，rows，cols）的4D图像张量，则应设置规范化的轴为1，即沿着通道轴规范化。输入格式是‘tf’同理。

- 1：按样本规范化，该模式默认输入为2D

我们大都使用的都是mode=0也就是按特征规范化，对于放置在卷积和池化之间或之后的4D张量，需要设置axis=1，而Dense层之后的BN层则直接使用默认值就好了。

**6、在验证集的误差不再下降时，如何中断训练？**

你可以使用 EarlyStopping 回调：

``` python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

**总结：关于callbacks参数的妙用**

（1）查询每隔epoch之后的loss和acc

（2）通过LearningRateScheduler实现衰减学习率或自定义衰减学习率

（3）通过EarlyStopping实现中断训练

（4）我们还可以自己定义回调函数，所为回调函数其实就是在训练完每一个epoch之后我们希望实现的操作。

**7.如何「冻结」网络层？**

「冻结」一个层意味着将其排除在训练之外，即其权重将永远不会更新。这在微调模型或使用固定的词向量进行文本输入中很有用。有两种方式实现：

**方式一：**在构造层的时候传递一个bool类型trainable参数，如下：

``` python
frozen_layer = Dense(32, trainable=False)
```

您可以将 trainable 参数（布尔值）传递给一个层的构造器，以将该层设置为不可训练的：

**方式二：**通过层对象的trainable属性去设置，如下：

```python
x = Input(shape=(32,))
layer = Dense(32)  #构造一个层
layer.trainable = False  #设置层的trainable属性
y = layer(x)
```

注意：可以在实例化之后将网络层的 trainable 属性设置为 True 或 False。为了使之生效，在修改 trainable 属性之后，需要在模型上调用 compile()。及重新编译模型。

**8.如何从 Sequential 模型中移除一个层？**

你可以通过调用模型的 .pop() 来删除 Sequential 模型中最后添加的层：

``` python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```

——from：[深度学习框架keras踩坑记](https://mp.weixin.qq.com/s/suBo64ozWDSu-rQv118IVA)







---

*update：2019-04-16* 









