## TensorFlow基础

### 一、TensorFlow基本概念

- 使用图（graphs）来表示计算任务
- 在被称之为会话（Session）的上下文（context）中执行图
- 使用张量（tensor）表示数据
- 通过变量（Variable）维护状态
- 使用 feed 和 fetch 可以为任意的操作赋值或者从其中获取数据

Tensorflow 是一个编程系统，使用图（graphs）来表示计算任务，图（graphs）中的节点称之为 op
（operation），一个 op 获得 0 个或多个 Tensor，执行计算，产生 0 个或多个 Tensor。Tensor 看作是
一个 n 维的数组或列表。图必须在会话（Session）里被启动。Tensorflow 结构如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-31113870.jpg)

关于张量（tensor）详细解释下：

> TensorFlow 中的所有数据如图片、语音等都是以张量这种数据结构的形式表示的。张量是一种组合类型的数据类型，表示为一个多维数组。
>
> 张量（tensor）的属性：维数（阶）、形状和数据类型。
>
> 张量的维数又叫张量的阶，是张量维数的一个数量描述。如下分别表示 0 维、1 维、2 维和 3 维的张量：
>
> ``` python
> 1    #维度为0的标量
> [1,2,3]   #维度为1, 一维向量
> [[1,2],[3,4]]   #维度为2, 二维矩阵
> [[[1,2],[3,4]],[[1,2],[3,4]]]   #维度为3, 3维空间矩阵
> ```
>
> 技巧：维度看张量的最左边有多少个左中括号，有 n 个，则这个张量就是 n 维张量。
>
> 张量的形状以  ***[D0, D1, … Dn-1]***  的形式表示，***D0*** 到 ***Dn*** 是任意的正整数。在运行程序查看结果常能注意到，比如：`shape=(1, 2)`，即表示形状为[1, 2]，第一维度有 1 个元素，第二维度 2 个元素——两个维度其实也就是矩阵了。
>
> 再如形状（注意：这里指的是张量形状）为[3, 4]表示第一维有 3 个元素，第二维有 4 个元素，而单纯[3, 4]表示一个 3 行 4 列的矩阵。
>
> ``` python
> 1    # 形状为[]
> [1,2,3]   # 形状为[3]
> [[1,2],[3,4]]   # 形状为[2,2]
> [[[1,2],[3,4]],[[1,2],[3,4]]]   # 形状为[2,2,2]
> ```

下面通过代码演示这个过程。

导入 tensorflow：

``` python
import tensorflow as tf
```

创建两个常量 op：

``` python
m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [3]])
```

创建一个矩阵乘法 op，把 m1 和 m2 传入：

``` python
product = tf.matmul(m1, m2)
print(product)
```

打印 product 结果如下：

``` xml
Tensor("MatMul:0", shape=(1, 1), dtype=int32)
```

可见，直接运行 tenserflow 中常量算术操作的得到的结果是一个张量。

接下来创建一个会话，启动默认图：

``` python
sess = tf.Session()
```

调用 sess 的 run 方法来执行矩阵乘法 op，run(product)触发了图中 3 个 op：

``` python
result = sess.run(product)
print(result)
sess.close()
```

打印结果如下：

``` xml
[[15]]
```

可见，真正要进行运算还需要使用会话操作。

PS：Session 最后需要关闭 `sess.close()`，以释放相关的资源。当然也可以使用`with`模块，session 在`with`模块中自动会关闭：

``` python
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
```

同样的打印结果如下：

``` xml
[[15]]
```

PS：关于 with，[Python 中 with用法及原理](https://blog.csdn.net/u012609509/article/details/72911564)

补充：TensorFlow 的这些节点最终将在计算设备（CPUs、GPus）上执行运算。如果是使用 GPU，默认会在第一块 GPU 上执行，如果想在第二块多余的 GPU 上执行：

``` python
with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...
```

device 中的各个字符串含义如下：

- `"/cpu:0"`：你机器的 CPU；
- `"/gpu:0"`：你机器的第一个 GPU；
- `"/gpu:1"`：你机器的第二个 GPU；

关于 tensor、operation、Session 到底该怎么去理解，我在网上找到了篇博客，值得看看：[tensorflow学习笔记（一）:基本知识之tensor，operation和Session](https://blog.csdn.net/woaidapaopao/article/details/72863591)，现摘录部分内容如下：

> tensor 可以理解为一种数据，TensorFlow 就相当于一个数据的流动过程，所有能用图（graph）来表示的计算任务理论上都能用 TensorFlow 来实现。有图就有节点和边的概念，其中节点是一种操作（如加减乘除）被称为 operation 简称 op，一个节点可以连接很多边，而边上传递的数据就是 tensor。为了能够实现一个计算流程就需要一个 graph 来表示这个流程，而为了能够执行这个图就需要一个会话（Session）来启动这个图，这个过程相当于 graph 是画好的一张图，然后我用一个 Session 来执行这个图。
>
> 由上面的过程就可以知道，想要实现一个计算就需要先画好一张图（graph），然后就是需要一个会话（Session）来启动这个图，然后通过 Session 的 run() 方法来计算图中需要计算的的值。最后就是对计算得到的值进行一些评估（这个比较复杂）。
>
> 过程就是：**建图 --> 启动图 --> 运行取值**。

### 二、TensorFlow使用变量

1）案例1：

``` python
import tensorflow as tf

# 定义变量x
x = tf.Variable([1, 2])
y = tf.constant([3, 3])

# 定义减法运算op
sub = tf.subtract(x, y)
# 定义加法运算op
add = tf.add(x, sub)

# 定义变量初始化器
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化所有变量
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))
```

运行结果如下：

``` xml
[-2 -1]
[-1  1]
```

需要注意的是，如果使用了变量，那么需要使用 `tf.global_variables_initializer()` 来初始化全局变量。

2）案例2：

``` python
import tensorflow as tf
# 给变量state起一个别名counter
state = tf.Variable(0, name='counter')
# state+1操作
new_value = tf.add(state, 1)
# 将new_value赋值给state
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    # 循环五次累加
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
```

运行结果如下：

``` xml
0
1
2
3
4
5
```

### 三、Fetch和Feed

#### Fetch指在一个会话中执行多个语句op：

``` python
import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    result = sess.run([mul, add])
    print(result)
```

运行结果如下：

``` xml
[21.0, 7.0]
```

其中语句：

``` python
result = sess.run([mul, add])
```

先执行了 mul 语句，之后再执行 add 语句，这便是 Feed。

#### Feed的数据以字典的形式传入：

``` python
import tensorflow as tf

# 创建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # Feed的数据以字典的形式传入
    print(sess.run(output, feed_dict={input1:[8.], input2:[2.]}))
```

运行结果如下：

``` xml
[ 16.]
```

程序开始创建的 input1 和 input2 两个占位符一开始只有类型没有赋值，然后再后面的运算中使用：

``` xml
feed_dict={input1:[8.], input2:[2.]}
```

给这两个占位符赋值并完成了后面的 output 运算。

### 四、使用Tensorflow完成梯度下降线性回归模型参数优化

``` python
import tensorflow as tf
import numpy as np

# 使用numpy随机生成100个点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

# 构建一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
# 定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数
train = optimizer.minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step, sess.run([k, b]))
```

运行结果如下：

``` xml
0 [0.050617315, 0.09909152]
20 [0.10102634, 0.19947496]
40 [0.10059734, 0.1996945]
60 [0.10034763, 0.1998222]
80 [0.100202315, 0.19989653]
100 [0.10011776, 0.19993977]
120 [0.10006853, 0.19996496]
140 [0.10003989, 0.1999796]
160 [0.100023225, 0.19998813]
180 [0.10001351, 0.19999309]
200 [0.10000786, 0.19999598]
```

设定的方程为 y=kx+b，其中参数 k = 0.1，b = 0.2，最后训练的结果和这个类似。