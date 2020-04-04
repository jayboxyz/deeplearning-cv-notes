## 交叉熵，过拟合，dropout以及TensorFlow中各种优化器的介绍

### 一、二次代价函数(quadratic cost)

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-97193262.jpg)

其中，C 表示代价函数，x 表示样本，y 表示实际值，a 表示输出值，n 表示样本的总数。为简单起见，我们以一个样本为例进行说明，此时二次代价函数为：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-95273637.jpg)

- a=σ(z), z=∑W j *X j +b

- σ() 是激活函数

假如我们使用梯度下降法(Gradient descent)来调整权值参数的大小，权值 w 和偏置 b 的梯度推导
如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-68276611.jpg)

其中，z 表示神经元的输入，σ 表示激活函数。w 和 b 的梯度跟激活函数的梯度成正比，激活函数的
梯度越大，w 和 b 的大小调整得越快，训练收敛得就越快。

假设我们的激活函数是 sigmoid 函数，其公式为：![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-88969788.jpg)

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-21437403.jpg)

- 假设我们的目标是收敛到 1，A 点为 0.82，距离目标比较远，梯度比较大，权值调整比较大；B 点为 0.98，距离目标比较近，梯度较小，权值调整比较小。调整方案合理。
- 假设我们的目标是收敛到 0，A 点为 0.82，距离目标比较近，梯度比较大，权值调整比较大；B 点为 0.98，距离目标比较远，梯度较小，权值调整比较小。调整方案不合理。

### 二、交叉熵代价函数(cross-entropy)

换一个思路，我们不改变激活函数，而是改变代价函数，改用交叉熵代价函数：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-99827572.jpg)

其中，C 表示代价函数，x 表示样本，y 表示实际值，a 表示输出值，n 表示样本的总数。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-69236364.jpg)

对 w 和 b 求偏导：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-31579009.jpg)

最后得出：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-58692742.jpg)

- 权值和偏置值的调整与![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-4716138.jpg)无关，另外，梯度公式中的![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-89981849.jpg)表示输出值与实
  际值的误差。所以当误差越大时，梯度就越大，参数 w 和 b 的调整就越快，训练的速度也就越快。
- 如果输出神经元是线性的，那么二次代价函数就是一种合适的选择。如果输出神经元是 S 型函数，
  那么比较适合用交叉熵代价函数。

### 三、对数释然代价函数(log-likelihood cost)

- 对数释然函数常用来作为 softmax 回归的代价函数，如果输出层神经元是 sigmoid 函数，可以采用
  交叉熵代价函数。而深度学习中更普遍的做法是将 softmax 作为最后一层，此时常用的代价函数是
  对数释然代价函数。
- 对数似然代价函数与 softmax 的组合和交叉熵与 sigmoid 函数的组合非常相似。对数释然代价函数
  在二分类时可以化简为交叉熵代价函数的形式。

在 Tensorflow 中用：

``` xml
tf.nn.sigmoid_cross_entropy_with_logits()来表示跟sigmoid搭配使用的交叉熵。
tf.nn.softmax_cross_entropy_with_logits()来表示跟softmax搭配使用的交叉熵。
```

测试如下：（对上次的手写数字识别代码修改，改为使用对数释然代价函数，对应代码：`4-1交叉熵.py`）

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
# loss = tf.reduce_mean(tf.square(y-prediction))
# 这里使用对数释然代价函数tf.nn.softmax_cross_entropy_with_logits()来表示跟softmax搭配使用的交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
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

运行结果如下：

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0,Testing Accuracy 0.8246
Iter 1,Testing Accuracy 0.8902
Iter 2,Testing Accuracy 0.9017
Iter 3,Testing Accuracy 0.9054
Iter 4,Testing Accuracy 0.9081
Iter 5,Testing Accuracy 0.9098
Iter 6,Testing Accuracy 0.9132
Iter 7,Testing Accuracy 0.9133
Iter 8,Testing Accuracy 0.9151
Iter 9,Testing Accuracy 0.9168
Iter 10,Testing Accuracy 0.9171
Iter 11,Testing Accuracy 0.9187
Iter 12,Testing Accuracy 0.9183
Iter 13,Testing Accuracy 0.9199
Iter 14,Testing Accuracy 0.9201
Iter 15,Testing Accuracy 0.9196
Iter 16,Testing Accuracy 0.921
Iter 17,Testing Accuracy 0.9211
Iter 18,Testing Accuracy 0.9208
Iter 19,Testing Accuracy 0.9214
Iter 20,Testing Accuracy 0.9217
```

下面将对比使用二次代价函数：

``` xml
对softmax使用二次代价函数结果		    	对softmax使用对数释然函数结果
Iter 0,Testing Accuracy 0.8313			Iter 0,Testing Accuracy 0.8246
Iter 1,Testing Accuracy 0.8703			Iter 1,Testing Accuracy 0.8902
Iter 2,Testing Accuracy 0.8813			Iter 2,Testing Accuracy 0.9017
Iter 3,Testing Accuracy 0.8876			Iter 3,Testing Accuracy 0.9054
Iter 4,Testing Accuracy 0.8938			Iter 4,Testing Accuracy 0.9081
Iter 5,Testing Accuracy 0.8976			Iter 5,Testing Accuracy 0.9098
Iter 6,Testing Accuracy 0.9002			Iter 6,Testing Accuracy 0.9132
Iter 7,Testing Accuracy 0.9013			Iter 7,Testing Accuracy 0.9133
Iter 8,Testing Accuracy 0.9043			Iter 8,Testing Accuracy 0.9151
Iter 9,Testing Accuracy 0.9056			Iter 9,Testing Accuracy 0.9168
Iter 10,Testing Accuracy 0.9064			Iter 10,Testing Accuracy 0.9171
Iter 11,Testing Accuracy 0.9066			Iter 11,Testing Accuracy 0.9187
Iter 12,Testing Accuracy 0.9082			Iter 12,Testing Accuracy 0.9183
Iter 13,Testing Accuracy 0.9095			Iter 13,Testing Accuracy 0.9199
Iter 14,Testing Accuracy 0.9096			Iter 14,Testing Accuracy 0.9201
Iter 15,Testing Accuracy 0.9109			Iter 15,Testing Accuracy 0.9196
Iter 16,Testing Accuracy 0.9128			Iter 16,Testing Accuracy 0.921
Iter 17,Testing Accuracy 0.9128			Iter 17,Testing Accuracy 0.9211
Iter 18,Testing Accuracy 0.9132			Iter 18,Testing Accuracy 0.9208
Iter 19,Testing Accuracy 0.9137			Iter 19,Testing Accuracy 0.9214
Iter 20,Testing Accuracy 0.9137			Iter 20,Testing Accuracy 0.9217              
```

可以看到，使用对数释然函数训练更快。

### 四、欠拟合、拟合、过拟合

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-54002718.jpg)

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-7703425.jpg)

防止过拟合：

1）增加数据集

一般来说，更多的数据参与训练得到的模型就越好，如果数据太少，而我们构建的神经网络又太复杂，节点很多的话就比较容易产生过拟合的现象。

2）正则化方法

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-50341618.jpg)

正则化方法是指在进行代价函数优化时，在代价函数后面加上一个正则项，这个正则项是跟权值相关的。入正则项系数，权衡正则项与 C0 的比重，n 是训练集样本的大小，它会使得原先那些处于 0 附近的权值往 0 移动，从而降低模型的复杂度，防止过拟合。

3）Dropout

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-33330943.jpg)

正则化方法通过在代价函数后面追加正则项来防止过度拟合的，还有一个方法是通过修改神经元本身的机构来实现的，称为 Dropout，其背后的情况是使部分神经元工作，部分神经元不工作。该方法是对神经网络进行训练时用到的一种技巧。

下面这个程序将演示过拟合和 Dropout 的效果。

还是以手写数字识别为例，与以前不同的是，这里构建的网络更加复杂，这里构建的输入层网络有 784 个神经元，第一个隐藏层有 2000 个神经元，第二个隐藏层有 2000 个神经元，第三个隐藏层有 1000 个神经元，输出层为 10 个神经元，这里的测试环境使用 GPU 版本，CPU 版本训练时间会过长。

第一轮实验使用所有的神经元进行训练：

``` python
sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
```

第二轮实验使用 70% 的神经元进行测试，也就是 Dropout：

``` python
sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
```

训练的代码如下：（对应代码：`4-2Dropout.py`）

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
keep_prob = tf.placeholder(tf.float32)

# 创建一个简单的神经网络
W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
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
    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))
```

使用全部神经元训练过程如下：（用的实验室电脑，显卡 GTX1080ti 跑的）

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0,Testing Accuracy 0.8581,Training Accuracy 0.86703634
Iter 1,Testing Accuracy 0.9595,Training Accuracy 0.97327274
Iter 2,Testing Accuracy 0.9619,Training Accuracy 0.98176366
Iter 3,Testing Accuracy 0.9658,Training Accuracy 0.98621815
Iter 4,Testing Accuracy 0.9664,Training Accuracy 0.9884545
Iter 5,Testing Accuracy 0.9679,Training Accuracy 0.99005455
Iter 6,Testing Accuracy 0.9672,Training Accuracy 0.9909818
Iter 7,Testing Accuracy 0.9692,Training Accuracy 0.99163634
Iter 8,Testing Accuracy 0.9698,Training Accuracy 0.9921455
Iter 9,Testing Accuracy 0.9699,Training Accuracy 0.99258184
Iter 10,Testing Accuracy 0.97,Training Accuracy 0.993
Iter 11,Testing Accuracy 0.97,Training Accuracy 0.9932909
Iter 12,Testing Accuracy 0.9705,Training Accuracy 0.99349093
Iter 13,Testing Accuracy 0.971,Training Accuracy 0.9937636
Iter 14,Testing Accuracy 0.9714,Training Accuracy 0.99416363
Iter 15,Testing Accuracy 0.9711,Training Accuracy 0.9943454
Iter 16,Testing Accuracy 0.9715,Training Accuracy 0.9945091
Iter 17,Testing Accuracy 0.9724,Training Accuracy 0.99465454
Iter 18,Testing Accuracy 0.9715,Training Accuracy 0.9948
Iter 19,Testing Accuracy 0.972,Training Accuracy 0.9948909
Iter 20,Testing Accuracy 0.9716,Training Accuracy 0.99496365
Iter 21,Testing Accuracy 0.972,Training Accuracy 0.99505454
Iter 22,Testing Accuracy 0.9718,Training Accuracy 0.9951636
Iter 23,Testing Accuracy 0.9716,Training Accuracy 0.9952
Iter 24,Testing Accuracy 0.972,Training Accuracy 0.9952545
Iter 25,Testing Accuracy 0.9722,Training Accuracy 0.9953455
Iter 26,Testing Accuracy 0.9725,Training Accuracy 0.9954364
Iter 27,Testing Accuracy 0.9723,Training Accuracy 0.9954909
Iter 28,Testing Accuracy 0.9724,Training Accuracy 0.9955636
Iter 29,Testing Accuracy 0.9731,Training Accuracy 0.9956727
Iter 30,Testing Accuracy 0.9729,Training Accuracy 0.9957455
```

Dropout后：（用的实验室电脑，显卡 GTX 1080ti 跑的）

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0,Testing Accuracy 0.9156,Training Accuracy 0.9118909
Iter 1,Testing Accuracy 0.9313,Training Accuracy 0.9254909
Iter 2,Testing Accuracy 0.9339,Training Accuracy 0.9346
Iter 3,Testing Accuracy 0.9387,Training Accuracy 0.9398182
Iter 4,Testing Accuracy 0.9445,Training Accuracy 0.9460727
Iter 5,Testing Accuracy 0.946,Training Accuracy 0.94805455
Iter 6,Testing Accuracy 0.9492,Training Accuracy 0.95149094
Iter 7,Testing Accuracy 0.9514,Training Accuracy 0.95461816
Iter 8,Testing Accuracy 0.9538,Training Accuracy 0.95716363
Iter 9,Testing Accuracy 0.9549,Training Accuracy 0.9582545
Iter 10,Testing Accuracy 0.9549,Training Accuracy 0.96009094
Iter 11,Testing Accuracy 0.9584,Training Accuracy 0.96114546
Iter 12,Testing Accuracy 0.9603,Training Accuracy 0.96312726
Iter 13,Testing Accuracy 0.9605,Training Accuracy 0.9649091
Iter 14,Testing Accuracy 0.9613,Training Accuracy 0.9654727
Iter 15,Testing Accuracy 0.9626,Training Accuracy 0.9674
Iter 16,Testing Accuracy 0.963,Training Accuracy 0.96754545
Iter 17,Testing Accuracy 0.9636,Training Accuracy 0.96876365
Iter 18,Testing Accuracy 0.9645,Training Accuracy 0.9701273
Iter 19,Testing Accuracy 0.9641,Training Accuracy 0.9696909
Iter 20,Testing Accuracy 0.9647,Training Accuracy 0.9710364
Iter 21,Testing Accuracy 0.9659,Training Accuracy 0.97136366
Iter 22,Testing Accuracy 0.9671,Training Accuracy 0.9731454
Iter 23,Testing Accuracy 0.9668,Training Accuracy 0.9734727
Iter 24,Testing Accuracy 0.9673,Training Accuracy 0.97374547
Iter 25,Testing Accuracy 0.9687,Training Accuracy 0.97465456
Iter 26,Testing Accuracy 0.9683,Training Accuracy 0.9756
Iter 27,Testing Accuracy 0.9695,Training Accuracy 0.9758545
Iter 28,Testing Accuracy 0.9717,Training Accuracy 0.9769818
Iter 29,Testing Accuracy 0.9711,Training Accuracy 0.9771636
Iter 30,Testing Accuracy 0.9701,Training Accuracy 0.97778183
```

可以对比两种训练方式最后的五次训练结果：

``` xml
# 使用所有神经元：
Iter 26,Testing Accuracy 0.9725,Training Accuracy 0.9954364
Iter 27,Testing Accuracy 0.9723,Training Accuracy 0.9954909
Iter 28,Testing Accuracy 0.9724,Training Accuracy 0.9955636
Iter 29,Testing Accuracy 0.9731,Training Accuracy 0.9956727
Iter 30,Testing Accuracy 0.9729,Training Accuracy 0.9957455

# Dropout:
Iter 26,Testing Accuracy 0.9683,Training Accuracy 0.9756
Iter 27,Testing Accuracy 0.9695,Training Accuracy 0.9758545
Iter 28,Testing Accuracy 0.9717,Training Accuracy 0.9769818
Iter 29,Testing Accuracy 0.9711,Training Accuracy 0.9771636
Iter 30,Testing Accuracy 0.9701,Training Accuracy 0.97778183
```

从上面可以看到，不使用 Dropout 方法训练后训练集测试网络准确率约 99%，测试集则为 97%，而且可以看出这个结果一直保持了很久，即使训练持续进行，这便是过拟合了。然而使用了 Dropout 方法训练后，测试集和训练集最后测试网络得到的结果基本差异不大，拟合度较高。

### 五、Optimizer优化器

Tensorflow 提供了下面这些种优化器：

- tf.train.GradientDescentOptimizer
- tf.train.AdadeltaOptimizer
- tf.train.AdagradOptimizer
- tf.train.AdagradDAOptimizer
- tf.train.MomentumOptimizer
- tf.train.AdamOptimizer
- tf.train.FtrlOptimizer
- tf.train.ProximalGradientDescentOptimizer
- tf.train.ProximalAdagradOptimizer
- tf.train.RMSPropOptimizer

各种优化器对比：

- 标准梯度下降法(**GD**, Gradient Descent)：标准梯度下降先计算所有样本汇总误差，然后根据总误差来更新权值

- 随机梯度下降法(**SGD**, Stochastic Gradient Descent)：随机梯度下降随机抽取一个样本来计算误差，然后更新权值

- 批量梯度下降法(**BGD**, Batch Gradient Descent)：批量梯度下降算是一种折中的方案，从总样本中选取一个批次（比如一共有 10000 个样本，随机选取 100 个样本作为一个 batch），然后计算这个 batch 的总误差，根据总误差来更新权值。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-17510205.jpg)

其中，

``` xml
W： 要训练的参数
J(W)： 代价函数
∇ W J(W)： 代价函数的梯度
η： 学习率
```

#### SGD：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-26957051.jpg)

#### Momentum：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-65038654.jpg)

当前权值的改变会受到上一次权值改变的影响，类似于小球向下滚动的时候带上了惯性。这样
可以加快小球的向下的速度。

#### NAG（Nesterov accelerated gradient）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-25559890.jpg)

NAG 在 TF 中跟 Momentum 合并在同一个函数 `tf.train.MomentumOptimizer` 中，可以通过参
数配置启用。

在 Momentun 中小球会盲目地跟从下坡的梯度，容易发生错误，所以我们需要一个更聪明的小球，这个小球提前知道它要去哪里，它还要知道走到坡底的时候速度慢下来而不是又冲上另一个坡。γvt−1 会用来修改W的值，计算 W−γvt−1 可以表示小球下一个位置大概在哪里。从而我们可以提前计算下一个位置的梯度，然后使用到当前位置。

#### Adagrad：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-53673825.jpg)

它是基于 SGD 的一种算法，它的核心思想是对比较常见的数据给予它比较小的学习率去调整
参数，对于比较罕见的数据给予它比较大的学习率去调整参数。它很适合应用于数据稀疏的数
据集（比如一个图片数据集，有 10000 张狗的照片，10000 张猫的照片，只有 100 张大象的照
片）。

Adagrad 主要的优势在于不需要人为的调节学习率，它可以自动调节。它的缺点在于，随着
迭代次数的增多，学习率也会越来越低，最终会趋向于 0。

#### RMSprop：

RMS（Root Mean Square）是均方根的缩写。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-58496947.jpg)

RMSprop 借鉴了一些 Adagrad 的思想，不过这里 RMSprop 只用到了前 t-1 次梯度平方的平均值加上当前梯度的平方的和的开平方作为学习率的分母。这样 RMSprop 不会出现学习率越来越低的问题，而且也能自己调节学习率，并且可以有一个比较好的效果。

#### Adadelta：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-379401.jpg)

使用 Adadelta 我们甚至不需要设置一个默认学习率，在 Adadelta 不需要使用学习率也可以达
到一个非常好的效果。

#### Adam：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-49863095.jpg)

就像 Adadelta 和 RMSprop 一样 Adam 会存储之前衰减的平方梯度，同时它也会保存之前衰减
的梯度。经过一些处理之后再使用类似 Adadelta 和 RMSprop 的方式更新参数。

关于优化器优缺点及如何选择网上有不错的总结：[关于深度学习优化器 optimizer 的选择，你需要了解这些](https://blog.csdn.net/g11d111/article/details/76639460)

下面使用 `tf.train.AdadeltaOptimizer` 来训练手写数字：（对应代码：`4-3优化器.py`）

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
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用AdamOptimizer
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

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

训练过程如下：（用的实验室电脑，显卡 GTX 1080ti 跑的）

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0,Testing Accuracy 0.9149
Iter 1,Testing Accuracy 0.9255
Iter 2,Testing Accuracy 0.9282
Iter 3,Testing Accuracy 0.9267
Iter 4,Testing Accuracy 0.9282
Iter 5,Testing Accuracy 0.9239
Iter 6,Testing Accuracy 0.9314
Iter 7,Testing Accuracy 0.931
Iter 8,Testing Accuracy 0.9299
Iter 9,Testing Accuracy 0.9282
Iter 10,Testing Accuracy 0.9318
Iter 11,Testing Accuracy 0.93
Iter 12,Testing Accuracy 0.9299
Iter 13,Testing Accuracy 0.9307
Iter 14,Testing Accuracy 0.9314
Iter 15,Testing Accuracy 0.9325
Iter 16,Testing Accuracy 0.9298
Iter 17,Testing Accuracy 0.9321
Iter 18,Testing Accuracy 0.9319
Iter 19,Testing Accuracy 0.9314
Iter 20,Testing Accuracy 0.9288
```


### 六、提高准确度

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
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype=tf.float32)

# 创建一个简单的神经网络
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# 交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 训练
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        learning_rate = sess.run(lr)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc) + ", Learning Rate= " + str(learning_rate))
```

训练过程如下：（用的实验室电脑，显卡 GTX 1080ti 跑的）

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0, Testing Accuracy= 0.9536, Learning Rate= 0.001
Iter 1, Testing Accuracy= 0.9608, Learning Rate= 0.00095
Iter 2, Testing Accuracy= 0.968, Learning Rate= 0.0009025
Iter 3, Testing Accuracy= 0.9713, Learning Rate= 0.000857375
Iter 4, Testing Accuracy= 0.9682, Learning Rate= 0.00081450626
Iter 5, Testing Accuracy= 0.9748, Learning Rate= 0.0007737809
Iter 6, Testing Accuracy= 0.9749, Learning Rate= 0.0007350919
Iter 7, Testing Accuracy= 0.9765, Learning Rate= 0.0006983373
Iter 8, Testing Accuracy= 0.9778, Learning Rate= 0.0006634204
Iter 9, Testing Accuracy= 0.9778, Learning Rate= 0.0006302494
Iter 10, Testing Accuracy= 0.9779, Learning Rate= 0.0005987369
Iter 11, Testing Accuracy= 0.9765, Learning Rate= 0.0005688001
Iter 12, Testing Accuracy= 0.9802, Learning Rate= 0.0005403601
Iter 13, Testing Accuracy= 0.9782, Learning Rate= 0.0005133421
Iter 14, Testing Accuracy= 0.9804, Learning Rate= 0.000487675
Iter 15, Testing Accuracy= 0.9803, Learning Rate= 0.00046329122
Iter 16, Testing Accuracy= 0.9797, Learning Rate= 0.00044012666
Iter 17, Testing Accuracy= 0.9814, Learning Rate= 0.00041812033
Iter 18, Testing Accuracy= 0.9812, Learning Rate= 0.00039721432
Iter 19, Testing Accuracy= 0.9815, Learning Rate= 0.0003773536
Iter 20, Testing Accuracy= 0.9802, Learning Rate= 0.00035848594
Iter 21, Testing Accuracy= 0.9816, Learning Rate= 0.00034056162
Iter 22, Testing Accuracy= 0.9818, Learning Rate= 0.00032353355
Iter 23, Testing Accuracy= 0.9817, Learning Rate= 0.00030735688
Iter 24, Testing Accuracy= 0.981, Learning Rate= 0.000291989
Iter 25, Testing Accuracy= 0.9812, Learning Rate= 0.00027738957
Iter 26, Testing Accuracy= 0.9808, Learning Rate= 0.0002635201
Iter 27, Testing Accuracy= 0.9814, Learning Rate= 0.00025034408
Iter 28, Testing Accuracy= 0.9815, Learning Rate= 0.00023782688
Iter 29, Testing Accuracy= 0.9814, Learning Rate= 0.00022593554
Iter 30, Testing Accuracy= 0.9809, Learning Rate= 0.00021463877
Iter 31, Testing Accuracy= 0.9822, Learning Rate= 0.00020390682
Iter 32, Testing Accuracy= 0.9823, Learning Rate= 0.00019371149
Iter 33, Testing Accuracy= 0.9824, Learning Rate= 0.0001840259
Iter 34, Testing Accuracy= 0.9822, Learning Rate= 0.00017482461
Iter 35, Testing Accuracy= 0.983, Learning Rate= 0.00016608338
Iter 36, Testing Accuracy= 0.9824, Learning Rate= 0.00015777921
Iter 37, Testing Accuracy= 0.9827, Learning Rate= 0.00014989026
Iter 38, Testing Accuracy= 0.9827, Learning Rate= 0.00014239574
Iter 39, Testing Accuracy= 0.9823, Learning Rate= 0.00013527596
Iter 40, Testing Accuracy= 0.9825, Learning Rate= 0.00012851215
Iter 41, Testing Accuracy= 0.9822, Learning Rate= 0.00012208655
Iter 42, Testing Accuracy= 0.9821, Learning Rate= 0.00011598222
Iter 43, Testing Accuracy= 0.9829, Learning Rate= 0.00011018311
Iter 44, Testing Accuracy= 0.9824, Learning Rate= 0.000104673956
Iter 45, Testing Accuracy= 0.9828, Learning Rate= 9.944026e-05
Iter 46, Testing Accuracy= 0.9829, Learning Rate= 9.446825e-05
Iter 47, Testing Accuracy= 0.9831, Learning Rate= 8.974483e-05
Iter 48, Testing Accuracy= 0.9827, Learning Rate= 8.525759e-05
Iter 49, Testing Accuracy= 0.9827, Learning Rate= 8.099471e-05
Iter 50, Testing Accuracy= 0.9828, Learning Rate= 7.6944976e-05
```



