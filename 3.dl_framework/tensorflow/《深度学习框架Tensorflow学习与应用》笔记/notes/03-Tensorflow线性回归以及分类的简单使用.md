## TensorFlow非线性回归以及分类的简单问题，softmax介绍

### 一、TensorFlow实现非线性回归

（对应代码：`3-1非线性回归.py`）

``` python
# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点，值在-0.5~0.5中，产生了200行一列的矩阵
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# 产生随机噪声
noise = np.random.normal(0, 0.02, x_data.shape)
# 给y_data加入噪声 y = x^2 + noise
y_data = np.square(x_data) + noise

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层,中间层权值为一行十列的矩阵
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
# 产生偏置值
biases_L1 = tf.Variable(tf.zeros([1, 10]))
# 预测结果：y = x * w + b
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# 使用tanh作为激活函数
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层，权值为十行一列的矩阵
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
```

运行结果如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-15616302.jpg)

### 二、TensorFlow解决手写数字识别（简单版本）

#### 1、MNIST数据集介绍

MNIST数据集可以从官网下载到本地：[Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)，也可以通过 Tensorflow 提供的 `input_data.py`进行载入。通过运行 Tensorflow 提供的代码加载数据集：

``` python
from tensorflow.examples.tutorials.mnist import input_data

# 获取数据
mnist = input_data.read_data_sets("D:/MNIST_data/", one_hot=True)
```

MNIST 数据集包含 55000 样本的训练集，5000 样本的验证集，10000 样本的测试集。 `input_data.py` 已经将下载好的数据集解压、重构图片和标签数据来组成新的数据集对象。

~~下载下来的数据集被分成两部分：60000 行的训练数据集（mnist.train）和 10000 行的测试数据集（mnist.test）。~~

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-21661076.jpg)

图像是`28x28`像素大小的灰度图片。我们把这一个数组展开成一个向量，长度是`28x28=784`。

~~因此在
 MNIST 训练数据集中 `mnist.train.images` 是一个形状为 [60000, 784] 的张量，第一个维度数字用
来索引图片，第二个维度数字用来索引每张图片中的像素点。图片里的某个像素的强度值介于 0-1 之间。~~

空白部分全部为 0，有笔迹的地方根据颜色深浅有 0~1 的取值，因此，每个样本有`28x28=784`维的特征，相当于展开为 1 维。

所以，训练集的特征是一个 55000x784 的 Tensor，第一纬度是图片编号，第二维度是图像像素点编号。而训练集的 `label`（图片代表的是 0~9 中哪个数）是一个 55000x10 的 Tensor，10 是 10 个种类的意思，进行 `one-hot 编码` 即只有一个值为 1，其余为 0，如数字 0，对于 label 为`[1,0,0,0,0,0,0,0,0,0]`。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-36607474.jpg)

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-96119059.jpg)

> MNIST 数据集的标签是介于 0-9 的数字，我们要把标签转化为“one-hot vectors”。一个 one-
> hot 向量除了某一位数字是 1 以外，其余维度数字都是 0，比如标签 0 将表示为`([1,0,0,0,0,0,0,0,0,0])`
> ，标签 3 将表示为`([0,0,0,1,0,0,0,0,0,0])` 。

~~因此， `mnist.train.labels` 是一个 [60000, 10] 的数字矩阵。~~

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-50265880.jpg)

综上，MNIST 的训练数据集是一个形状为~~`60000x784`的 tensor~~，也就是一个多维数组，第一维表示图片的索引，第二维表示图片中像素的索引。

MNIST 中的数字手写体图片的 label 值在 0 到 9 之间，是图片所表示的真实数字。这里用 One-hot vector 来表述 label值，vector 的长度为 label 值的数目，vector 中有且只有一位为 1，其他为 0，为了方便，我们表示某个数字时在 vector 中所对应的索引位置设置 1，其他位置元素为 0，例如用 `[0,0,0,1,0,0,0,0,0,0]` 来表示 3。所以， mnist.train.labels是一个~~`60000x10`的二维数组。~~   

#### 2、神经网络构建

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-25158586.jpg)

#### 3、Softmax函数

我们知道 MNIST 的结果是 0-9，我们的模型可能推测出一张图片是数字 9 的概率是80%，是数字 8
 的概率是 10%，然后其他数字的概率更小，总体概率加起来等于 1。这是一个使用 softmax 回归模
型的经典案例。softmax 模型可以用来给不同的对象分配概率。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-96955391.jpg)

比如输出结果为[1,5,3]：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-57443260.jpg)

#### 4、编码实现

（对应代码：`3-2MNIST数据集分类简单版本.py`）

``` python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
```

在我笔记本上运行结果如下：

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0,Testing Accuracy 0.8313
Iter 1,Testing Accuracy 0.8703
Iter 2,Testing Accuracy 0.8813
Iter 3,Testing Accuracy 0.8876
Iter 4,Testing Accuracy 0.8938
Iter 5,Testing Accuracy 0.8976
Iter 6,Testing Accuracy 0.9002
Iter 7,Testing Accuracy 0.9013
Iter 8,Testing Accuracy 0.9043
Iter 9,Testing Accuracy 0.9056
Iter 10,Testing Accuracy 0.9064
Iter 11,Testing Accuracy 0.9066
Iter 12,Testing Accuracy 0.9082
Iter 13,Testing Accuracy 0.9095
Iter 14,Testing Accuracy 0.9096
Iter 15,Testing Accuracy 0.9109
Iter 16,Testing Accuracy 0.9128
Iter 17,Testing Accuracy 0.9128
Iter 18,Testing Accuracy 0.9132
Iter 19,Testing Accuracy 0.9137
Iter 20,Testing Accuracy 0.9137
```

#### 5、代码讲解

##### Softmax Regression的程序实现

关于载入数据集代码`mnist = input_data.read_data_sets("MNIST_data", one_hot=True)`补充下：

1. 第一个参数直接填写文件夹名称，则表示使用的为当前程序路径，可以改为其他目录，比如`D:\\mnist_data\\`

2. 下载后的数据集如下：

   ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-52299788.jpg)

   如果下载不下来，可以网上搜索单独下载保存到本地。

为了使用 TensorFlow，我们需要引用该库函数

``` python
import tensorflow as tf
```

我们利用一些符号变量来描述交互计算的过程，创建如下

``` python
x = tf.placeholder(tf.float32, [None, 784])
```

这里的 `x` 不是一个特定的值，而是一个占位符，即需要时指定。我们用一个`1x784`维的向量来表示一张 MNIST 中的图片。我们用`[None, 784]`这样一个二维的 tensor 来表示整个 MNIST 数据集，其中`None`表示可以为任意值。

我们使用`Variable`(变量)来表示模型中的权值和偏置，这些参数是可变的。如下，

``` python
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

这里的 W 和 b 均被初始化为 0 值矩阵。W的维数为`784x10`，是因为我们需要将一个 784 维的像素值经过相应的权值之乘转化为 10 个类别上的 evidence 值；b 是十个类别上累加的偏置值。

实现 softmax regression 模型仅需要一行代码，如下：

``` python
prediction = tf.nn.softmax(tf.matmul(x, W) + b)
```

其中，`matmul`函数实现了 x 和 W 的乘积，这里 x 为二维矩阵，所以放在前面。可以看出，在 TensorFlow 中实现 softmax regression 模型是很简单的。

##### 模型的训练

在机器学习中，通常需要选择一个代价函数（或者损失函数），来指示训练模型的好坏。

``` python
# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
```

接下来我们以代价函数最小化为目标，来训练模型以得到相应的参数值(即权值和偏置)。TensorFlow 知道你的计算过程，它会自动利用反向传播算法来得到相应的参数变化，对代价函数最小化的影响作用。然后，你可以选择一个优化算法来决定如何最小化代价函数。如下，

``` python
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
```

在这里，我们使用了一个学习率为 0.2 的梯度下降算法来最小化代价函数。梯度下降是一个简单的计算方式，即使得变量值朝着减小代价函数值的方向变化。TensorFlow 也提供了许多[其他的优化算法](https://www.tensorflow.org/versions/master/api_docs/python/train.html#optimizers)，仅需要一行代码即可实现调用。

TensorFlow 提供了以上简单抽象的函数调用功能，你不需要关心其底层实现，可以更加专心于整个计算流程。在模型训练之前，还需要对所有的参数进行初始化：

``` python
# 初始化变量
init = tf.global_variables_initializer()
```

我们可以在一个 Session 里面运行模型，并且进行初始化：

``` python
with tf.Session() as sess:
    sess.run(init)
    ...
```

接下来，进行模型的训练：

```python
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
```

每一次的循环中，我们取训练数据中的 batch_size(100) 个随机数据，这种操作成为批处理(batch)。然后，每次运行 train_step 时，将之前所选择的数据，填充至所设置的占位符中，作为模型的输入。

以上过程成为随机梯度下降，在这里使用它是非常合适的。因为它既能保证运行效率，也能一定程度上保证程序运行的正确性。（理论上，我们应该在每一次循环过程中，利用所有的训练数据来得到正确的梯度下降方向，但这样将非常耗时）。

##### 模型的评价

怎样评价所训练出来的模型？显然，我们可以用图片预测类别的准确率。

首先，利用`tf.argmax()`函数来得到预测和实际的图片 label 值，再用一个`tf.equal()`函数来判断预测值和真实值是否一致。如下：

``` python
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
```

correct_prediction 是一个布尔值的列表，例如 [True, False, True, True]。可以使用`tf.cast()`函数将其转换为[1, 0, 1, 1]，以方便准确率的计算。

``` python
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

最后，我们来获取模型在测试集上的准确率，并打印：

``` python
acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
```

代码的讲解参考：[TensorFlow学习笔记1：入门](http://www.jeyzhang.com/tensorflow-learning-notes.html)