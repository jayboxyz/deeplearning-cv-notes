## RNN（递归神经网络）和LSTM（长短期记忆网络）

### RNN（Recurrent Neural Network）

相关资料：

- [深度学习笔记——RNN（LSTM、GRU、双向RNN）学习总结](https://blog.csdn.net/mpk_no1/article/details/72875185)
- 莫烦：[什么是循环神经网络 RNN (Recurrent Neural Network) ](https://morvanzhou.github.io/tutorials/machine-learning/keras/2-4-A-RNN/)
- [递归神经网络](https://feisky.xyz/machine-learning/rnn/)
- ......

递归神经网络（Recurrent Neural Networks，RNN）是两种人工神经网络的总称：时间递归神经网络（recurrent neural network）和结构递归神经网络（recursive neural network）。时间递归神经网络的神经元间连接构成有向图，而结构递归神经网络利用相似的神经网络结构递归构造更为复杂的深度网络。

RNN 一般指代时间递归神经网络。单纯递归神经网络因为无法处理随着递归，权重指数级爆炸或消失的问题（Vanishing gradient problem），难以捕捉长期时间关联；而结合不同的 LSTM 可以很好解决这个问题。时间递归神经网络可以描述动态时间行为，因为和前馈神经网络（feedforward neural network）接受较特定结构的输入不同，RNN 将状态在自身网络中循环传递，因此可以接受更广泛的时间序列结构输入。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-10-88546789.jpg)

从基础的神经网络中知道，神经网络包含输入层、隐层、输出层，通过激活函数控制输出，层与层之间通过权值连接。激活函数是事先确定好的，那么神经网络模型通过训练“学“到的东西就蕴含在“权值“中。 

基础的神经网络只在层与层之间建立了权连接，RNN 最大的不同之处就是在层之间的神经元之间也建立的权连接，如图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-10-65343447.jpg)

这是一个标准的 RNN 结构图，图中每个箭头代表做一次变换，也就是说箭头连接带有权值。左侧是折叠起来的样子，右侧是展开的样子，左侧中 h 旁边的箭头代表此结构中的“循环“体现在隐层。 

在展开结构中我们可以观察到，在标准的 RNN 结构中，隐层的神经元之间也是带有权值的。也就是说，随着序列的不断推进，前面的隐层将会影响后面的隐层。图中 O 代表输出，y 代表样本给出的确定值，L 代表损失函数，我们可以看到，“损失“也是随着序列的推荐而不断积累的。 

除上述特点之外，标准 RNN 的还有以下特点： 

1. 权值共享，图中的 W 全是相同的，U 和 V 也一样。 
2. 每一个输入值都只与它本身的那条路线建立权连接，不会和别的神经元连接。

以上是 RNN 的标准结构，然而在实际中这一种结构并不能解决所有问题。RNN 还有很多种结构，用于应对不同的需求和解决不同的问题。

### LSTM（Long Short-Term Memory Network）

相关资料：

- [推荐给初学LSTM或者懂个大概却不完全懂的人](https://blog.csdn.net/roslei/article/details/61912618)
- 莫烦：[LSTM RNN 循环神经网络 (LSTM)](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/2-4-LSTM/)
- ......

LSTM 全称叫 Long Short-Term Memory networks，它和传统 RNN 唯一的不同就在与其中的神经元（感知机）的构造不同。传统的 RNN 每个神经元和一般神经网络的感知机没啥区别，但在 LSTM 中，每个神经元是一个“记忆细胞”（元胞状态，Cell State），将以前的信息连接到当前的任务中来。每个 LSTM 细胞里面都包含：

- 输入门（input gate）
- 遗忘门（forget gate)
- 输出门（output gate）

如何理解？相关资料的第一篇文章提到三个例子，来看看：

1）遗忘门

> 作用对象：细胞状态 
>
> 作用：将细胞状态中的信息选择性的遗忘 
>
> 让我们回到语言模型的例子中来基于已经看到的预测下一个词。在这个问题中，细胞状态可能包含当前主语的类别，因此正确的代词可以被选择出来。当我们看到新的主语，我们希望忘记旧的主语。 
>
> 例如，他今天有事，所以我。。。当处理到‘’我‘’的时候选择性的忘记前面的’他’，或者说减小这个词对后面词的作用。

2）输入门

> 作用对象：细胞状态
>
> 作用：将新的信息选择性的记录到细胞状态中
>
> 在我们语言模型的例子中，我们希望增加新的主语的类别到细胞状态中，来替代旧的需要忘记的主语。 
> 例如：他今天有事，所以我。。。。当处理到‘’我‘’这个词的时候，就会把主语我更新到细胞中去。 

3）输出门

> 作用对象：隐层 ht 
>
> 在语言模型的例子中，因为他就看到了一个 代词，可能需要输出与一个 动词 相关的信息。例如，可能输出是否代词是单数还是负数，这样如果是动词的话，我们也知道动词需要进行的词形变化。 
>
> 例如：上面的例子，当处理到‘’我‘’这个词的时候，可以预测下一个词，是动词的可能性较大，而且是第一人称。 会把前面的信息保存到隐层中去。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-10-64259652.jpg)

注：GRU（Gated Recurrent Unit）就是 LSTM 的一个变态，这是由 Cho, et al. (2014) 提出。它将忘记门和输入门合成了一个单一的更新门。同样还混合了细胞状态和隐藏状态，和其他一些改动。最终的模型比标准的 LSTM 模型要简单，也是非常流行的变体。

### 测试代码（例子：手写数字识别MNIST）

> 这节来讲下 LSTM 在 TensorFlow 中的实现。LSTM 主要用在语言、文本等序列化问题上，但是呢，同样地也可以用在图像上，做图像分类其实也是可以的。下面通过简单例子来讲解下是怎么实现的。

（对应代码：`7-2递归神经网络RNN.py`，下面代码导入了 rnn，使用了 tf1.0 之后版本 LSTM 基本 CELL）

``` python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# 输入图片是28*28
n_inputs = 28  # 输入一行，一行有28个数据
max_time = 28  # 一共28行
lstm_size = 100  # 隐层单元
n_classes = 10  # 10个分类
batch_size = 50  # 每批次50个样本
n_batch = mnist.train.num_examples // batch_size  # 计算一共有多少个批次

# 这里的none表示第一个维度可以是任意的长度
x = tf.placeholder(tf.float32,[None,784])
# 正确的标签
y = tf.placeholder(tf.float32,[None,10])

# 初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))
```



``` python
# 定义RNN网络
def RNN(X,weights,biases):
    # inputs=[batch_size, max_time, n_inputs]
    inputs = tf.reshape(X,[-1,max_time,n_inputs])
    # 定义LSTM基本CELL
    # lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)  # 老版本
    lstm_cell = rnn.BasicLSTMCell(lstm_size)  # 1.0之后版本
    # final_state[0]是cell state
    # final_state[1]是hidden_state
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1],weights) + biases)
    return results
```



``` python
# 计算RNN的返回结果
prediction = RNN(x, weights, biases)
# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))# argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))# 把correct_prediction变为float32类型
# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print ("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))
```

测试过程如下：

``` xml
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Iter 0, Testing Accuracy= 0.702
Iter 1, Testing Accuracy= 0.7982
Iter 2, Testing Accuracy= 0.82
Iter 3, Testing Accuracy= 0.8317
Iter 4, Testing Accuracy= 0.9021
Iter 5, Testing Accuracy= 0.9171
```

