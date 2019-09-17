## 一、TensorFlow基础：TensorFlow的编程思想

TensorFlow 的编程要按照一定规则来进行。

在 TensorFlow 程序中应包含两个部分：一个是构建计算图的部分，另一个是把建好的计算图放在一个 Sesstion 会话中的执行部分。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190118160643.png)

- 构建计算图：这是定义变量、初始化数据及建立运算关系的一个过程。TensorFlow 把这样一个过程称为构建计算图。计算图(Graph)由节点(operation)和边(Tensor、Variable)组成。其中节点是各种 operation 操作，比如程序中的矩阵乘法`tf.matmul`，边是运算数据或变量，在 TensorFlow 中称为 tensor。举个例子：比如我们要编写 tf 程序实现`y=w*x`，那么我首先要构建一个图，这个图只包括一个节点，就是矩阵乘法操作，然后输入有两个边分别是 x 和 w，如上面的图所示。
- 把计算图放到一个Sesstion中执行：构建好图(Graph)之后，要先将这个 graph 添加到一个在会话`tf.session`里面，并使用`tf.Session().run()方法`运行计算图。

首先给一个 hello world 的 tensorflow 程序：实现矩阵乘法`y = W*x`，其中，

``` xml
w = [[3.0,2.5],[1.0,2.7]]； 
x = [1.0,1.0] 
```

程序如下：

``` python
import tensorflow as tf

###-----必须先构建计算图（y=w*x）---------###
graph1 = tf.Graph()                        #定义图graph1
with graph1.as_default():                  #指定图 graph1为默认图
    w = tf.Variable([[3.0,2.5],[1.0,2.7]]) #图中边Variabel的定义
    x = tf.constant(1.0,shape=[1,2])       #图中边x_tensor的生成

    y = tf.matmul(w,tf.transpose(x))       #节点：变量的乘法和转置操作（operation）

    init_op = tf.global_variables_initializer()   #节点： Variable的initialize_operation

###------建好的计算图放在一个Sesstion中执行-------###
with tf.Session(graph=graph1) as sess:    # 打开Session，把graph1添加到默认会话sess中
    sess.run(init_op)                     # 执行init_op操作，此时tf.matmul并没有run  
    print(sess.run(y))                    # 在会话中执行tf.matmul操作, 打印y_tensor
```

注：

- tensor：可以理解为一个多维数组，是 tensor 中的数据形式，类似于 numpy 中的 narray，例如下图我们输入数据集 X，以及输出 y。
- Operation：：比如矩阵乘法操作。
- 必须注意的是，构建好的图，必须在会话 Session 中使用 run( ) 方法，才能实现 tensor 运算。

总结：

- Tensorflow 程序是以计算图 Graph 为基础的计算单元，并且计算图需要在一个指定的会话 Sesstion 中执行。
- 一个计算图（Graph）由操作(operation)构成的节点以及 tensor数据流/variable变量数据构成的边组成。换句话说，图中的节点(圆圈)就是一些操作（Operation），比如加、减、乘、除等等。而节点之间的边就是张量流（Tensor-flow）或变量 Variable。
- 计算图必须被放进一个会话 Sesstion 里面，并使用 Sesstion 的 run 方法执行计算图的某个 operation，这个 operation 才会被执行。而其他没有被 run 的 operation，仍然不会被执行。



## 二、TensorFlow基础：Graph计算图的创建和使用

在【TensorFlow的编程思想】说到，在 tensorflow 程序中应包含两个部分：一个是构建计算图的部分，另一个是把建好的计算图放在一个 Sesstion 会话中的执行部分。 这篇主要讲使用`tf.Graph()`函数创建一个新的计算图的方法。

**(1) 只创建一个Graph图**

在 Tensorflow 中，始终存在一个默认的 Graph，当你创建 Operation、Tensor 时，tensorflow 会将你这些节点和边自动添加到这个默认 Graph 中。 

那么，当你只想创建一个图时，并不需要使用`tf.Graph()`函数创建一个新图，而是直接定义需要的 Tensor 和Operation，这样，tensorflow 会将这些节点和边自动添加到默认 Graph 中。

``` python
import tensorflow as tf

###-----图的构建阶段---------###
w = tf.Variable([[3.0,2.5],[1.0,2.7]]) #图中边Variabel的定义
x = tf.constant(1.0,shape=[1,2])       #图中边x_tensor的生成

y = tf.matmul(w,tf.transpose(x))       #节点：变量的乘法和转置操作（operation）
init_op = tf.global_variables_initializer()   #节点:Variable的initialize_operation
#----------tensorflow会将上面定义的节点和边自动添加到默认Graph中------#

###------图的执行阶段-------###
with tf.Session() as sess:    
    sess.run(init_op)      # 执行init_op操作，此时tf.matmul并没有run  
    print(sess.run(y))     # 在会话中执行tf.matmul操作, 打印y_tensor，
```

**(2) 定义多个Graph图**

在 tensorflow 中，可以使用`tf.Graph()`函数创建图。如果我们需要定义多个 Graph，则可以在 with 语句中调用`tf.Graph.as_default()`方法将某个 graph 设置成默认 Graph，这样 with 语句块中调用的 Operation或 Tensor 将会添加到该 Graph 中。

with 语句是保证操作的资源可以正确的打开和释放，而且不同的计算图上的张量和运算彼此分离，互不干扰。

``` python
import tensorflow as tf
graph1 = tf.Graph()
with graph1.as_default():#创建图1
    c1 = tf.constant([9.0])

with tf.Graph().as_default() as graph2:  #创建图2
    c2 = tf.constant([1.0])

with tf.Session(graph=graph1) as sess1:#使用sess1 运行graph1
    print (sess1.run(c1))
with tf.Session(graph=graph2) as sess2:#使用sess2 运行graph2
    print (sess2.run(c2))
```

注：定义多个图时，可以通过设置`tf.Sesstion(graph=)` 中的参数，**选择当前的Session执行哪个计算图**。

**(3) 指定Graph计算图运行的设备**

TensorFlow 中计算图可以通过`tf.Graph.device`函数来指定**运行计算图的设备**。下面程序将加法计算放在 CPU 上执行，也可以使用`tf.device`达到同样的效果。

``` python
g = tf.Graph()
with g.device('/cpu:0'):
    result = 1 + 2
```



## 三、TensorFlow基础：tensor张量、tensor的属性、tensor数据和numpy数据的转化

在【TensorFlow的编程思想】说到，计算图(Graph)由节点(operation)和边(Tensor、Variable)组成。其中节点是各种 operation 操作，边是运算数据或变量，在 tensorflow 中称为 tensor，那么什么是 tensor 呢？它的有什么属性？

**(1) 什么是tensor张量**

tensor 是 tensorflow 中的数据形式。是一种可以表示多维数组的 class 类，可以理解为多维数组。

**(2) tensor的属性**

在 tensor 类中包含以下几个属性：

- name 属性：name是一个Tensor的唯一标识符。

  - 如果我们没有指定 name 的值，则 tensorflow 会按操作名自动分配 name 值，比如用`a = tf.contstant(1.0)`定义一个 tensor 常量 a，则 tensorflow 会将 name 设置为`a.name = "Const:0"`。
  - Tensor 的 name 属性可以通过 noedName_k:src_output 形式给出，例如"Mul_9:0"。 
    - 其中，nodeName=Mul 表示为乘法操作；
    - k=9 表示为第 10 个同名的 Mul 操作，在 tensorflow 中，当我们对两个 tensor 指定同一个 name 值时，tensorflow 会自动加`_k`加以区分；
    - src_output=0 表示当前节点的第 0 个输出。

- shape 属性：描述维数信息。

- dtype 属性：tensor 的数据类型，tensorflow 会对所有参与计算的 Tensor 进行类型检查，当发现类型不匹配时会报错，例如下面程序中 b 改为`dtype= tf.float64`，则会报错。 

  ``` python
  import tensorflow as tf
  a = tf.constant([1.0,2.0],name='A'，dtype= tf.float32) 
  b = tf.constant([2.0,3.0],name='A'，dtype= tf.float32) #指定同一个name，tf会自动加_k加以区分
  r = tf.add(a,b)#没指定name，默认为操作名
  
  print(a)
  print(b) #打印tensor
  print(r)
  
  out:
  Tensor("A:0", shape=(2,), dtype=float32)
  Tensor("A_1:0", shape=(2,), dtype=float32)
  Tensor("Add:0", shape=(2,), dtype=float32)
  ```

**(3) numpy数据和tensor数据转换**

tensor 其实是一种可以表示多维数组的 class 类，和 numpy 可以互相转化。 

函数形式：`tf.convert_to_tensor(arr)`

``` python
import tensorflow as tf
import numpy as np

arr = np.ones([2,3])
print(type(arr))

tensor = tf.convert_to_tensor(arr,name='x')  # ndarrray ---->tensor
print(type(tensor))

with tf.Session() as sess:
    print(sess.run(tensor))
```

补充：在 tensorflow 的类似 tf.constant() 的方法的参数可以是 List（列表，如：`[2, 3, 5]`）、ndarry（多维数组， eg：`[[2, 3], [4, 5]]`）。



## 四、TensorFlow基础：Sesstion会话

在【TensorFlow的编程思想】中，我们说到每个计算图都必须要在一个会话 Session 中执行。本节主要讲会话的作用以及怎么使用会话。

**(1) 会话(Session)的作用**

会话(Session)可以管理 TensorFlow 运行时的所有资源。 计算图都是在 Sesstion 中运行的，因此会话中拥有很多资源，并且 Sesstion 可以对这些资源进行管理。例如，当所有计算完成后，可以使用 Session.close() 释放会话资源，这样的资源管理可以避免资源浪费。

**(2) 会话的生成方式**

主要有：

1. 函数生成法 `tf.sesstion()`
2. 上下文管理器 `with tf.sesstion() as sess`
3. 默认会话 `with sess.as_default()`
4. 交互式会话 `tf.InteractiveSession()`

①使用 tf.sesstion() 函数生成法：

``` python
#创建一个会话
sess = tf.Session()
#使用这个会话可以得到张量的结果,例如sess.run(result)
sess.run(op)
#关闭会话
sess.close()
```

注：最后需要加上` sess.close()`来关闭会话。

②使用上下问管理器：

``` python
#创建一个会话,通过上下文管理器管理会话
with tf.Session() as sess:
    sess.run(op)
```

③使用默认会话：

在前面计算图我们知道，可以指定某个计算图为默认图，同样的，我们用函数`tf.session（）`生成的 sess，可以使用`sess.as_default()`方法手动指定默认会话。

``` python
a = tf.constant([1.0])
b = tf.constant([2.0])
output = a + b
sess = tf.Session()
with sess.as_default():
    print(result.eval()) #计算张量的结果
```

④使用交互式会话：

交互式环境下，比如 iPython，直接使用`tf.InteractiveSession()` 构造默认会话。那么Tensor.eval() 和 Operation.run() 方法会使用这个默认会话去执行操作。

``` python
sess=tf.InteractiveSession()
a=tf.constant(5)
b=tf.Variable(3)

init_op = tf.global_variables_initializer() #变量同样需要先初始化
c=tf.multiply(a,b)

sess.run(init_op)
print (sess.run(c))
```

**(3) tf.InteractiveSession()与tf.Session()的区别**

tf.InteractiveSession() 实际上构建了一个默认会话，且 Tensor.eval() 和 Operation.run() 方法会使用这个默认会话去执行操作 run ops。

- tf.Session()需要在启动session之前**先构建整个计算图**，然后启动该计算图。
- tf.InteractiveSession()可以**先构建一个session然后再定义操作**（operation），主要最后要`sess.close()`。



## 五、TensorFlow基础：tensor常量生成

tensorflow 中的 tensor 常量：常量在深度学习中，经常用于变量 Variable 的初始化，常用的常量有随机常量、常数常量、全 0、全 1常量。

**(1) 随机常量**

在神经网络中，经常需要用随机常量来初始化一些变量，例如我们在初始化**权重向量W**时，经常要使用**正态分布随机初始化。**

①正态分布

**函数形式：**

``` xml
tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,,name=None))
tf.truncated_normal(shape,mean=0.0,stddev=1.0,dtype = tf.float32,,name=None))
```

其中，参数含义：张量形状，平均值，标准差，数据类型。

两个正态分布的区别是： 

- `tf.random_normal`是普通的正态分布函数。 
- `tf.truncated_normal`截断的正态分布函数。其生成的值遵循一个正态分布，**但如果生成但随机值偏离平均值超过 2 个标准差，这个数会被重新随机分配。**

②均匀分布

**函数形式：**

``` xml
tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32)
```

其参数的含义是：形状，最小值，最大取值，取值类型等。

③洗牌

经常用于讲训练集洗牌，避免同一分类下的样本紧紧挨在一起。

函数形式：

``` xml
tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32)
```

其参数含义是：形状，最小值，最大取值，取值类型等。

``` xml
tf.random_shuffle(shape,minval=0,maxval=None,dtype=tf.float32)
```

**(2) 常数常量**

①`tf.constant(obj,shape)`

其中，obj 可以是  list 或者常数，生成各种类型的 tensor。

``` python
a = tf.constant(np.arange(1, 13, dtype=np.int32),shape=[2, 2, 3])  
```

②`tf.linspace(start, end, num)`

其中，start 代表起始的值，end 表示结束的值，num 表示在这个区间里生成数字的个数，生成的数组是等间隔生成的。start 和 end 这两个数字必须是浮点数，不能是整数，如果是整数会出错的。

③`tf.range(start, end, alpha)`

其中，start 代表起始的值，end 表示结束的值，alpha 表示步长。

**(3) 全0、全1常量**

①生成与 tensor 相同 shape 的全 0，全1 tensor 矩阵

- `tf.zeros_like(tensor)`
- `tf.ones_like(tensor)`

②全 0，全1 的 tensor 矩阵

- `tf.zeros(shape,dtype)`
- `tf.ones(shape,dtype)`



## 六、TensorFlow基础：tensor变量 tf.Variable与tf.get_variable和tf.variable_scope

**(1) tf.Variable与tf.get_variable创建变量**

`tf.Variable`与`tf.get_variable`都可以在 tensorflow **定义变量**，当他们用来创建变量时，他们的区别在于：

- `tf.Variable`的变量名是一个 可选项。但是`tf.get_variable`必须指定变量名。
- `tf.get_variable`一旦指定了一个变量名，就不能再重复定义。除非结合`tf.variable_scope`中的`reuse`参数。`tf.Variable`用相同 name 参数指定两个变量是不会报错的。

``` python
v1 = tf.get_variable('v', shape=[1], initializer=tf.constant_initializer(1.0))
v2 = tf.get_variable('v', shape=[1], initializer=tf.constant_initializer(1.0))

ValueError: Variable v already exists, disallowed.

v1 = tf.Variable(tf.random_normal(shape=[2,2]), name='v')
v2 = tf.Variable(tf.random_normal(shape=[2,2]), name='v')
```

不会报错。

**函数定义格式**如下：

①`tf.Variable(init_obj, name='v')`用于生成一个初始值为 init-obj 的变量。

- init_obj 为必须项，它是变量的初始化数据，一般对权重变量初始化采用正态随机初始化。
- name是一个可选项。

②`tf.get_variabl(name, shape=None, dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None))`获取已存在的变量， 不存在则新建一个变量。

- name是一个必要的参数选项
- 变量的初始化可以利用 initializer 来实现。比如： [Xavier初始化器](https://blog.csdn.net/promisejia/article/details/81635830)。

``` python
#变量创建的等价定义
v = tf.get_variable('v', shape=[1], initializer=tf.constant_initializer(1.0))
v = tf.Variable(tf.random_normal(shape=[2,2]), name='v')
```

**(2) tf.variable_scope()与tf.get_variable的配合使用**

`tf.variable_scope(name,resue=False)`与`tf.get_variable`经常配合使用，更加方便地**管理参数命名**。

上面说到，tf.get_variable 一旦指定了一个变量名，就不能再用该变量名重复定义。但是在神经网络中我们第一层和第二层的参数都可以称为 weight时，就不可以直接使用 tf.get_variable，而是和要结合 tf.variable_scope() 定义不同的命名空间将两种变量区别开来。

``` python
with tf.variable_scope('layer1',resue=False):
    v = tf.get_variable('v',[1],initializer=tf.constant_initializer(1.0))

with tf.variable_scope('layer2',resue=False):
    v1 = tf.get_variable('v',[1])
```

另外，还必须知道：

- 当 reuse 为 False 或者 None 时（这也是默认值），同一个 tf.variable_scope 下面的变量名不能相同；
- 当 reuse 为 True 时，tf.variable_scope 只能获取已经创建过的变量。

违反上面两个情况都会报错。

**(3) 使用tf.get_variable的好处**

- 可以使用 reuse 参数，公共同一命名空间下的变量；
- 可以和`tf.variable_scope`结合，管理变量。

---

*注：全文来源于 CSDN 博主[promisejia](https://blog.csdn.net/promisejia)文章。*



## 其他阅读：

- [【TensorFlow动手玩】基本概念: Tensor, Operation, Graph](https://blog.csdn.net/shenxiaolu1984/article/details/52813962)
- [TensorFlow基础（一）: tensor and operation](https://www.jianshu.com/p/6fec37e6ccc1)
- [tensorflow-基础知识 - 执剑长老 - 博客园](https://www.cnblogs.com/qjoanven/p/7736025.html)
- [TensorFlow学习（二）：变量常量类型](https://blog.csdn.net/xierhacker/article/details/53103979)
- ……

