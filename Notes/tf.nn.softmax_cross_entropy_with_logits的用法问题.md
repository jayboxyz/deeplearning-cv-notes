

发现个问题：在【[04-softmax，交叉熵(cross-entropy)，dropout以及Tensorflow中各种优化器的介绍](./Notes/04-softmax，交叉熵\(cross-entropy\)，dropout以及Tensorflow中各种优化器的介绍.md)】中（三）节开始的代码`4-1交叉熵.py`，可以注意到如下代码：

``` python
# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 这里使用对数释然代价函数tf.nn.softmax_cross_entropy_with_logits()来表示跟softmax搭配使用的交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
```

我觉得这里是有问题的，问题在哪呢？先看[【TensorFlow】tf.nn.softmax_cross_entropy_with_logits的用法](https://blog.csdn.net/zj360202/article/details/78582895)该文，可以了解到 `tf.nn.softmax_cross_entropy_with_logits` 函数的 logits 参数传入的是未经过 softmax 的 label 值。

``` python
import tensorflow as tf  

#our NN's output  
logits=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])  
```

``` python
#step1:do softmax  
y=tf.nn.softmax(logits)  
#true label  
y_=tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])  
#step2:do cross_entropy  
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  
```

两步可以用这一步代替：

``` python
#do cross_entropy just one step  
cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, y_))#dont forget tf.reduce_sum()!! 
```

但`prediction = tf.nn.softmax(tf.matmul(x, W) + b)`这里的 prediction 已经经历了一次 softmax，然后又经过了 tf.nn.softmax_cross_entropy_with_logits 函数，这相当于经过两个 softmax 了（虽然大的值的概率值还是越大，这点上倒是没影响。）

关于 softmax_cross_entropy_with_logits，这篇文章也有提到【[TensorFlow入门](https://statusrank.xyz/articles/31693f5f.html)】：

> 这个函数内部自动计算 softmax 函数，然后在计算代价损失,所以logits必须是未经 softmax 处理过的函数，否则会造成错误。

注1：好像后面的笔记中程序代码都是这样的，我觉得可能视频讲解老师没注意到这点问题。另外，在该文 [06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题](./Notes/06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题.md) 的笔记中，我也记录了该问题。

注2：对 softmax、softmax loss、cross entropy 不了解，推荐看这几篇：**[卷积神经网络系列之softmax，softmax loss和cross entropy的讲解](https://blog.csdn.net/u014380165/article/details/77284921)**  | [CNN入门讲解：我的Softmax和你的不太一样](<https://zhuanlan.zhihu.com/p/41784404>)，讲解的非常易懂。【荐】

另外，关于在 TensorFlow 中有哪些损失函数的实现呢？看看该文：[tensorflow API:tf.nn.softmax_cross_entropy_with_logits()等各种损失函数](https://blog.csdn.net/NockinOnHeavensDoor/article/details/80139685)