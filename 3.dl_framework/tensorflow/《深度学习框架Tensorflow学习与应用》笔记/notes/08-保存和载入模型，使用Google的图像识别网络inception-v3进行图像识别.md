##  保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别

### 一、保存和载入模型

#### 1、保存模型

可以使用：

``` python
saver = tf.train.Saver()
saver.save() 
```

来保存模型。

完整代码如下：（对应代码：`8-1saver_save.py`）

``` python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次100张照片
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

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

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(11):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
    # 保存模型
    saver.save(sess, 'net/my_net.ckpt')
```

上面定义了一个 saver：

``` python
saver = tf.train.Saver()
```

训练结束了使用：

``` python
saver.save(sess, 'net/my_net.ckpt')
```

将训练好的模型保存在 net/my_net.ckpt 文件中。

训练过程如下：

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0,Testing Accuracy 0.8239
Iter 1,Testing Accuracy 0.8893
Iter 2,Testing Accuracy 0.9001
Iter 3,Testing Accuracy 0.9051
Iter 4,Testing Accuracy 0.9081
Iter 5,Testing Accuracy 0.9094
Iter 6,Testing Accuracy 0.9112
Iter 7,Testing Accuracy 0.9132
Iter 8,Testing Accuracy 0.9142
Iter 9,Testing Accuracy 0.9158
Iter 10,Testing Accuracy 0.9171
```

最后 net 目录下有如下文件：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-10-50408346.jpg)

#### 2、载入模型

可以使用该方式来调用一个训练好的模型：

``` python
saver = tf.train.Saver()
saver.restore()
```

案例完整代码如下：（对应代码：`8-2saver_restore.py`）

``` python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次100张照片
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    saver.restore(sess,'net/my_net.ckpt')
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
```

测试结果如下：

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
0.098
0.9166
```

如上使用了：

``` python
saver = tf.train.Saver()
saver.restore(sess,'net/my_net.ckpt')
```

来调用上节训练好的手写数字识别模型。代码做了个测试，一开始直接将测试集送往没有训练好的网络，得到的测试结果是 0.098，然后调用训练好的网络，测试结果为 0.9166。

#### 3、补充内容（加载预训练模型和保存模型，以及fine-tuning）

先看看：[TensorFlow学习笔记（8）--网络模型的保存和读取](https://blog.csdn.net/lwplwf/article/details/62419087)，自己手敲一遍。

**（1）保存TensorFlow模型：** 

``` python
import tensorflow as tf

# 声明两个变量
v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
init_op = tf.global_variables_initializer() # 初始化全部变量
saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
with tf.Session() as sess:
    sess.run(init_op)
    print("v1:", sess.run(v1)) # 打印v1、v2的值一会读取之后对比
    print("v2:", sess.run(v2))
    saver_path = saver.save(sess, "D:/save/model.ckpt")  # 将模型保存到D盘save/model.ckpt文件
    print("Model saved in file:", saver_path)
```

> **!!!顺便补充**：在常见的神经网络代码中，如果想要在 1000 次迭代后，再保存模型，只需设置`global_step`参数即可
>
> ``` python
> saver.save(sess, './checkpoin t_dir/MyModel',global_step=1000)
> ```
>
> 保存的模型文件名称会在后面加`-1000`。关于 global_step 的理解，一起来看下该文里的讲解：[TensorFlow学习笔记：Saver与Restore](https://www.jianshu.com/p/b0c789757df6)，在看完文章，并自己亲自实践后，以下是我的理解，先看代码：
>
> ``` python
> import tensorflow as tf
> v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
> init_op = tf.global_variables_initializer()
> saver = tf.train.Saver(max_to_keep=3)
> with tf.Session() as sess:
>     for epoch in range(1, 8):
>         sess.run(init_op)
>         print("v1:", sess.run(v1))
>         saver_path = saver.save(sess, "D:/logs/model.ckpt", global_step=epoch)
> ```
>
> 运行上面代码后，会自动生成最近 3 个 ckpt 文件（因为max_to_keep=3），所以你可以看到：
>
> ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181207215549.png)
>
> 然后`saver.restore(sess, "D:/11hzb/model.ckpt-6")`恢复的为 epoch 为 6 的那次的参数值，即倒数第二次的参数值。
>
> 然后代码改为：
>
> ``` python
> saver = tf.train.Saver(max_to_keep=3)
> with tf.Session() as sess:
>     for epoch in range(1, 8):
>         sess.run(init_op)
>         print("v1:", sess.run(v1))
>         saver_path = saver.save(sess, "D:/logs/model.ckpt", global_step=2)
> ```
>
> 或是改为 global_step=3，或是 global_step=4 等等…本地只会生成对应的一个 ckpt 文件，名称为`model.ckpt-x`格式，其中 x 为 global_step 的值。
>
> 然后恢复`saver.restore(sess, "D:/11hzb/model.ckpt-2")`，可以看到都是打印的最后一次的模型保存的 v1 的参数值。我在理解是：每次运行到这行`saver_path = saver.save(sess, "D:/11hzb/model.ckpt", global_step=2)`，名称都是这个名称 model.ckpt-2 文件，第二次运行到这里也是这个名称，覆盖上一次；再运行到这，再去覆盖……最后实质保存的其实就是最后一次的参数值。
>
> 在实际训练中，我们可能会在每 1000 次迭代中保存一次模型数据，但是由于图是不变的，没必要每次都去保存，可以通过如下方式指定不保存图：
>
> ``` python
> if step % 1000 == 0:
>     saver.save(sess, './checkpoint_dir/MyModel',global_step=step,write_meta_graph=False)
> ```
>
> 另一种比较实用的是，如果你希望每 2 小时保存一次模型，并且只保存最近的 5 个模型文件：
>
> ``` python
> tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
> ```
>
> 注意：tensorflow 默认只会保存最近的 5 个模型文件，如果你希望保存更多，可以通过`max_to_keep`来指定。
>
> 如果我们不对 tf.train.Saver 指定任何参数，默认会保存所有变量。如果你不想保存所有变量，而只保存一部分变量，可以通过指定 variables/collections。在创建 tf.train.Saver 实例时，通过将需要保存的变量构造 list 或者 dictionary，传入到 Saver 中：
>
> ``` python
> import tensorflow as tf
> w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
> w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
> saver = tf.train.Saver([w1,w2])
> sess = tf.Session()
> sess.run(tf.global_variables_initializer())
> saver.save(sess, './checkpoint_dir/MyModel',global_step=1000)
> ```
>
> 更多内容阅读：[Tensorflow加载预训练模型和保存模型](https://blog.csdn.net/huachao1001/article/details/78501928)，该文还介绍了下 fine-tuning（微调） ，建议看下。
>
> PS：代码中 `saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))` 的 latest_checkpoint 函数为用来自动获取最后一次保存的模型。

运行结果：

``` xml
v1: [[-0.22841574  0.6937564 ]]
v2: [[-0.05212444  0.29719114 -0.31847867]
 [ 0.7853754  -1.4160358   0.5617032 ]]
Model saved in file: D:/save/model.ckpt
```

且在 D 盘 save 文件夹下可以看到如下文件：（程序结束后，会生成四个文件：存储网络结构 `.meta`、存储训练好的参数 `.data` 和 `.index`、记录最新的模型 checkpoint。*From：[【tensorflow】保存模型、再次加载模型等操作](https://blog.csdn.net/liuxiao214/article/details/79048136)*）

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181206175132.png)

> 这段代码中，通过`saver.save`函数将 TensorFlow 模型保存到了 save/model.ckpt 文件中，这里代码中指定路径为`"D:/save/model.ckpt"`，也就是保存到了 D 盘里面的`save`文件夹中。
>
> TensorFlow 模型会保存在后缀为`.ckpt`的文件中。保存后在 save 这个文件夹中会出现 3 个文件，因为TensorFlow会将计算图的结构和图上参数取值分开保存。
>
> - checkpoint 
>
>   checkpoint 文件保存了一个目录下所有的模型文件列表，这个文件是 tf.train.Saver 类自动生成且自动维护的。在 checkpoint 文件中维护了由一个 tf.train.Saver 类持久化的所有 TensorFlow 模型文件的文件名。当某个保存的 TensorFlow 模型文件被删除时，这个模型所对应的文件名也会从 checkpoint 文件中删除。checkpoint 中内容的格式为 CheckpointState Protocol Buffer。
>
>   **总结：**checkpoint 文件是个文本文件，里面记录了保存的最新的 checkpoint 文件以及其它 checkpoint 文件列表。在 inference 时，可以通过修改这个文件，指定使用哪个 model。
>
> - model.ckpt.meta 
>
>   model.ckpt.meta 文件保存了 TensorFlow 计算图的结构，可以理解为神经网络的网络结构。TensorFlow 通过元图（MetaGraph）来记录计算图中节点的信息以及运行计算图中节点所需要的元数据。TensorFlow 中元图是由 MetaGraphDef Protocol Buffer 定义的。MetaGraphDef 中的内容构成了 TensorFlow 持久化时的第一个文件。保存 MetaGraphDef 信息的文件默认以 .meta 为后缀名，文件 model.ckpt.meta 中存储的就是元图数据。
>
>   **总结：**meta 文件保存的是图结构，是 pb（protocol buffer）格式文件，包含变量、op、集合等。
>
> - model.ckpt
>
>   model.ckpt 文件保存了 TensorFlow 程序中每一个变量的取值，这个文件是通过 SSTable 格式存储的，可以大致理解为就是一个（key，value）列表。model.ckpt 文件中列表的第一行描述了文件的元信息，比如在这个文件中存储的变量列表。列表剩下的每一行保存了一个变量的片段，变量片段的信息是通过 SavedSlice Protocol Buffer 定义的。SavedSlice 类型中保存了变量的名称、当前片段的信息以及变量取值。TensorFlow 提供了 tf.train.NewCheckpointReader 类来查看 model.ckpt 文件中保存的变量信息。如何使用 tf.train.NewCheckpointReader 类这里不做说明，自查。
>
>   **总结：**ckpt 文件是二进制文件，保存了所有的 weights、biases、gradients 等变量。在 tensorflow  0.11 之前，保存在`.ckpt`文件中。0.11 后，通过两个文件保存，如：
>
>   ``` xml
>   model.ckpt.data-00000-of-00001
>   model.ckpt.index
>   ```

**（2）加载TensorFlow模型：**

①方法一：

``` python
import tensorflow as tf

# 使用和保存模型代码中一样的方式来声明变量
v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
with tf.Session() as sess:
    saver.restore(sess, "save/model.ckpt") # 即将固化到硬盘中的Session从保存路径再读取出来
    print("v1:", sess.run(v1)) # 打印v1、v2的值和之前的进行对比
    print("v2:", sess.run(v2))
    print("Model Restored")
```

运行结果：

``` xml
v1: [[-0.22841574  0.6937564 ]]
v2: [[-0.05212444  0.29719114 -0.31847867]
 [ 0.7853754  -1.4160358   0.5617032 ]]
Model Restored
```

这段加载模型的代码基本上和保存模型的代码是一样的。也是先定义了 TensorFlow 计算图上所有的运算，并声明了一个 tf.train.Saver 类。两段唯一的不同是，在加载模型的代码中没有运行变量的初始化过程，而是将<u>**变量的值通过已经保存的模型加载进来**</u>。也就是说使用 TensorFlow 完成了一次模型的保存和读取的操作。

> 在恢复的代码中，图中的变量什么的都差不多(和保存模型来对比)，但是这段代码中没有变量的初始化过程，这里需要注意的是，变量的值是通过已经保存的模型加载进来。变量名不需要一模一样，但是名字"name"需要一样，变量的初始值形状一样就行。也就是说，最后从保存的模型中恢复的时候，是按照 name 参数的名字来对应找的。
> 然而这样的方式是重复定义了计算图上面的基本运算。你必须定义和原来的一 样的代码才能够得到存储的东西,使用非常受限制，在一些简单的地方使用这个方式是很好的。 
>
> 还有一种方法就是**不重新定义图的运算，直接加载已经持久化的图**。这种方法更加灵活，但是也有点小复杂。
>
> *From：[模型的保存与恢复（上）基本操作](https://blog.csdn.net/xierhacker/article/details/58637829)* 

②方法二

如果不希望重复定义图上的运算，也可以直接加载已经持久化的图：

``` python
import tensorflow as tf
# 在下面的代码中，默认加载了TensorFlow计算图上定义的全部变量
# 直接加载持久化的图
saver = tf.train.import_meta_graph("save/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "save/model.ckpt")
    # 通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))
```

运行结果：

``` xml
[[-0.22841574  0.6937564 ]]
```

**有时可能只需要保存或者加载部分变量。** 比如，可能有一个之前训练好的 5 层神经网络模型，但现在想写一个 6 层的神经网络，那么可以将之前 5 层神经网络中的参数直接加载到新的模型，而仅仅将最后一层神经网络重新训练。

为了保存或者加载部分变量，在声明`tf.train.Saver`类时可以提供一个列表来指定需要保存或者加载的变量。比如在加载模型的代码中使用`saver = tf.train.Saver([v1])`命令来构建`tf.train.Saver`类，那么只有变量 v1 会被加载进来。

另外再补充些网上博文，多了解下：*From [TensorFlow模型保存和提取方法](https://blog.csdn.net/marsjhao/article/details/72829635)*

> 1. TensorFlow 通过 tf.train.Saver 类实现神经网络模型的保存和提取。tf.train.Saver 对象 saver 的 save 方法将 TensorFlow 模型保存到指定路径中，saver.save(sess,"Model/model.ckpt")，实际在这个文件目录下会生成 4 个文件。
>
>    ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181206194841.png)
>
>    checkpoint 文件保存了一个录下多有的模型文件列表，model.ckpt.meta 保存了 TensorFlow 计算图的结构信息，model.ckpt 保存每个变量的取值，此处文件名的写入方式会因不同参数的设置而不同，但加载 restore 时的文件路径名是以 checkpoint 文件中的“model_checkpoint_path”值决定的。
>
> 2. 加载这个已保存的 TensorFlow 模型的方法是 `saver.restore(sess,"./Model/model.ckpt")`，加载模型的代码中也要定义 TensorFlow 计算图上的所有运算并声明一个 tf.train.Saver 类，不同的是加载模型时不需要进行变量的初始化，而是将变量的取值通过保存的模型加载进来，注意加载路径的写法。若不希望重复定义计算图上的运算，可直接加载已经持久化的图，`saver =tf.train.import_meta_graph("Model/model.ckpt.meta")`。
>
> 3. tf.train.Saver 类也支持在保存和加载时给变量重命名，声明 Saver 类对象的时候使用一个字典 dict 重命名变量即可，{"已保存的变量的名称name": 重命名变量名}，`saver = tf.train.Saver({"v1":u1, "v2": u2})`即原来名称 name 为 v1 的变量现在加载到变量 u1（名称 name 为 other-v1）中。
>
> 4. 上一条做的目的之一就是方便使用变量的滑动平均值。如果在加载模型时直接将影子变量映射到变量自身，则在使用训练好的模型时就不需要再调用函数来获取变量的滑动平均值了。载入时，声明 Saver 类对象时通过一个字典将滑动平均值直接加载到新的变量中，saver = tf.train.Saver({"v/ExponentialMovingAverage": v})，另通过 tf.train.ExponentialMovingAverage 的 `variables_to_restore()` 函数获取变量重命名字典。
>
>    此外，通过 convert_variables_to_constants 函数将计算图中的变量及其取值通过常量的方式保存于一个文件中。
>
> TensorFlow程序实现
>
> ``` python
> # 本文件程序为配合教材及学习进度渐进进行，请按照注释分段执行
> # 执行时要注意IDE的当前工作过路径，最好每段重启控制器一次，输出结果更准确
>  
>  
> # Part1: 通过tf.train.Saver类实现保存和载入神经网络模型
>  
> # 执行本段程序时注意当前的工作路径
> import tensorflow as tf
>  
> v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
> v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
> result = v1 + v2
>  
> saver = tf.train.Saver()
>  
> with tf.Session() as sess:
>     sess.run(tf.global_variables_initializer())
>     saver.save(sess, "Model/model.ckpt")
>  
>  
> # Part2: 加载TensorFlow模型的方法
>  
> import tensorflow as tf
>  
> v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
> v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
> result = v1 + v2
>  
> saver = tf.train.Saver()
>  
> with tf.Session() as sess:
>     saver.restore(sess, "./Model/model.ckpt") # 注意此处路径前添加"./"
>     print(sess.run(result)) # [ 3.]
>  
>  
> # Part3: 若不希望重复定义计算图上的运算，可直接加载已经持久化的图
>  
> import tensorflow as tf
>  
> saver = tf.train.import_meta_graph("Model/model.ckpt.meta")
>  
> with tf.Session() as sess:
>     saver.restore(sess, "./Model/model.ckpt") # 注意路径写法
>     print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))) # [ 3.]
>  
>  
> # Part4： tf.train.Saver类也支持在保存和加载时给变量重命名
>  
> import tensorflow as tf
>  
> # 声明的变量名称name与已保存的模型中的变量名称name不一致
> u1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
> u2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
> result = u1 + u2
>  
> # 若直接生命Saver类对象，会报错变量找不到
> # 使用一个字典dict重命名变量即可，{"已保存的变量的名称name": 重命名变量名}
> # 原来名称name为v1的变量现在加载到变量u1（名称name为other-v1）中
> saver = tf.train.Saver({"v1": u1, "v2": u2})
>  
> with tf.Session() as sess:
>     saver.restore(sess, "./Model/model.ckpt")
>     print(sess.run(result)) # [ 3.]
>  
>  
> # Part5: 保存滑动平均模型
>  
> import tensorflow as tf
>  
> v = tf.Variable(0, dtype=tf.float32, name="v")
> for variables in tf.global_variables():
>     print(variables.name) # v:0
>  
> ema = tf.train.ExponentialMovingAverage(0.99)
> maintain_averages_op = ema.apply(tf.global_variables())
> for variables in tf.global_variables():
>     print(variables.name) # v:0
>                           # v/ExponentialMovingAverage:0
>  
> saver = tf.train.Saver()
>  
> with tf.Session() as sess:
>     sess.run(tf.global_variables_initializer())
>     sess.run(tf.assign(v, 10))
>     sess.run(maintain_averages_op)
>     saver.save(sess, "Model/model_ema.ckpt")
>     print(sess.run([v, ema.average(v)])) # [10.0, 0.099999905]
>  
>  
> # Part6: 通过变量重命名直接读取变量的滑动平均值
>  
> import tensorflow as tf
>  
> v = tf.Variable(0, dtype=tf.float32, name="v")
> saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
>  
> with tf.Session() as sess:
>     saver.restore(sess, "./Model/model_ema.ckpt")
>     print(sess.run(v)) # 0.0999999
>  
>  
> # Part7: 通过tf.train.ExponentialMovingAverage的variables_to_restore()函数获取变量重命名字典
>  
> import tensorflow as tf
>  
> v = tf.Variable(0, dtype=tf.float32, name="v")
> # 注意此处的变量名称name一定要与已保存的变量名称一致
> ema = tf.train.ExponentialMovingAverage(0.99)
> print(ema.variables_to_restore())
> # {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
> # 此处的v取自上面变量v的名称name="v"
>  
> saver = tf.train.Saver(ema.variables_to_restore())
>  
> with tf.Session() as sess:
>     saver.restore(sess, "./Model/model_ema.ckpt")
>     print(sess.run(v)) # 0.0999999
>  
>  
> # Part8: 通过convert_variables_to_constants函数将计算图中的变量及其取值通过常量的方式保存于一个文件中
>  
> import tensorflow as tf
> from tensorflow.python.framework import graph_util
>  
> v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
> v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
> result = v1 + v2
>  
> with tf.Session() as sess:
>     sess.run(tf.global_variables_initializer())
>     # 导出当前计算图的GraphDef部分，即从输入层到输出层的计算过程部分
>     graph_def = tf.get_default_graph().as_graph_def()
>     output_graph_def = graph_util.convert_variables_to_constants(sess,
>                                                         graph_def, ['add'])
>  
>     with tf.gfile.GFile("Model/combined_model.pb", 'wb') as f:
>         f.write(output_graph_def.SerializeToString())
>  
>  
> # Part9: 载入包含变量及其取值的模型
>  
> import tensorflow as tf
> from tensorflow.python.platform import gfile
>  
> with tf.Session() as sess:
>     model_filename = "Model/combined_model.pb"
>     with gfile.FastGFile(model_filename, 'rb') as f:
>         graph_def = tf.GraphDef()
>         graph_def.ParseFromString(f.read())
>  
>     result = tf.import_graph_def(graph_def, return_elements=["add:0"])
>     print(sess.run(result)) # [array([ 3.], dtype=float32)]
> 
> ```
>
>



### 二、使用Google的图像识别网络inception-v3进行图像识别

先了解下 inception 网络模型，参考博客：

- [TensorFlow学习笔记：使用Inception v3进行图像分类](https://www.jianshu.com/p/cc830a6ed54b)
- [Google Inception Net介绍及Inception V3结构分析](https://blog.csdn.net/weixin_39881922/article/details/80346070)
- [深入浅出——网络模型中Inception的作用与结构全解析](https://blog.csdn.net/u010402786/article/details/52433324)
- [tensorflow+inceptionv3图像分类网络结构的解析与代码实现【附下载】](https://blog.csdn.net/k87974/article/details/80221215)
- ......

#### 1、下载inception-v3网络模型

（对应代码：`8-3下载google图像识别网络inception-v3并查看结构.py`）

``` py
# coding: utf-8

import tensorflow as tf
import os
import tarfile
import requests

# inception模型下载地址
inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# 模型存放地址
inception_pretrain_model_dir = "inception_model"
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)

# 获取文件名，以及文件路径
filename = inception_pretrain_model_url.split('/')[-1]
filepath = os.path.join(inception_pretrain_model_dir, filename)

# 下载模型
if not os.path.exists(filepath):
    print("download: ", filename)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("finish: ", filename)
# 解压文件
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)

# 模型结构存放文件
log_dir = 'inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# classify_image_graph_def.pb为google训练好的模型
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
with tf.Session() as sess:
    # 创建一个图来存放google训练好的模型
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    # 保存图的结构
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()
```

在 Jupyter Notebook 中运行代码后显示：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-10-50884014.jpg)

然后在相应目录会出现如下两个文件夹：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-10-14887577.jpg)

其中，inception_log 文件夹保存模型的结构：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-10-84328307.jpg)

inception_model 文件夹下是保存的训练结果：（其他文件其实都是`inception-2015-12-05.tgz`文件解压后的）

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-10-79742604.jpg)

其中，`classify_image_graph_def.pb`是已经训练过的 inception-v3 的模型。

#### 2、使用inception-v3网络模型进行图像识别

我们先打开 inception_model 文件夹下的 `imagenet_2012_challenge_label_map_proto.pbtxt` 和 `imagenet_synset_to_human_label_map.txt` 看看。

两个文件内容如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-10-98506442.jpg)

简单说明：左侧文件中 target_class 后面的数字代表目标的分类，数值为 1——1000（inception 模型是用来做 1000 个分类的），target_class_string 后面的字符串值对应到右侧文件的第一列，右侧文件的第二列表示对第一列的描述，相当是对分类的描述，从而来表示属于哪一类。

在运行代码之前，先在在当前程序路径下新建 images 文件夹，网上找几张图片保存在 images 下。

完整代码如下：（对应代码：`8-4使用inception-v3做各种图像的识别.py`）

``` python
# coding: utf-8

import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

class NodeLookup(object):
    def __init__(self):  
        label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'   
        uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 加载分类字符串n********对应分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        #一行一行读取数据
        for line in proto_as_ascii_lines :
            #去掉换行符
            line=line.strip('\n')
            #按照'\t'分割
            parsed_items = line.split('\t')
            #获取分类编号
            uid = parsed_items[0]
            #获取分类名称
            human_string = parsed_items[1]
            #保存编号字符串n********与分类名称映射关系
            uid_to_human[uid] = human_string  # n00004475->organism, being

        # 加载分类字符串n********对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                #获取分类编号1-1000
                target_class = int(line.split(': ')[1])  # target_class: 449
            if line.startswith('  target_class_string:'):
                #获取编号字符串n********
                target_class_string = line.split(': ')[1]  # target_class_string: "n01440764"
                #保存分类编号1-1000与编号字符串n********映射关系
                node_id_to_uid[target_class] = target_class_string[1:-2]  # 449->n01440764

        #建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            #获取分类名称
            name = uid_to_human[val]
            #建立分类编号1-1000到分类名称的映射关系
            node_id_to_name[key] = name  # 449->organism, being
        return node_id_to_name

    #传入分类编号1-1000返回分类名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]
```



``` python
#创建一个图来存放google训练好的模型
with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
```



``` python
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    #遍历目录
    for root,dirs,files in os.walk('images/'):
        for file in files:
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root,file), 'rb').read()
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})#图片格式是jpg格式
            predictions = np.squeeze(predictions)#把结果转为1维数据

            #打印图片路径及名称
            image_path = os.path.join(root,file)
            print(image_path)
            #显示图片
            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            #排序
            top_k = predictions.argsort()[-5:][::-1]
            print('top_k:', top_k)
            node_lookup = NodeLookup()
            for node_id in top_k:     
                #获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                #获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()
```

代码中，程序的头读取了两个文件：

``` xml
    label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'   
    uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
```

代码中，类  `NodeLookup` 的目的就是建立两个文件之间的关系，将`imagenet_2012_challenge_label_map_proto.pbtxt`中的 target_class 对应于`imagenet_synset_to_human_label_map.txt`中的类。

最后的排序代码解释下：

``` python
			#排序
            top_k = predictions.argsort()[-5:][::-1]
            print('top_k:', top_k)
            node_lookup = NodeLookup()
            for node_id in top_k:     
                #获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                #获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()
```

因为概率从小到大排序，所以如上第一行代码表示从倒数第 5 的位置开始取至倒数第 1 的位置，从而得到概率顺序从小到大的前 5 的概率值，再对这 5 个值做个倒序，进而得到从大到小的 5 个概率值。

最后的运行结果如下：

``` xml
images/lion.jpg
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-34173500.jpg)

``` xml
top_k: [190  11 206  85  30]
lion, king of beasts, Panthera leo (score = 0.96306)
cougar, puma, catamount, mountain lion, painter, panther, Felis concolor (score = 0.00161)
cheetah, chetah, Acinonyx jubatus (score = 0.00079)
leopard, Panthera pardus (score = 0.00057)
jaguar, panther, Panthera onca, Felis onca (score = 0.00033)
```

``` xml
images/panda.jpg
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-5249455.jpg)

``` xml
top_k: [169   7 222 374 878]
giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca (score = 0.96960)
lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens (score = 0.00078)
soccer ball (score = 0.00067)
lawn mower, mower (score = 0.00065)
earthstar (score = 0.00040)
```

``` xml
images/rabbit.jpg
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-48396384.jpg)

``` xml
top_k: [164 840 129 950 188]
Angora, Angora rabbit (score = 0.36784)
hamper (score = 0.17425)
hare (score = 0.13834)
shopping basket (score = 0.10668)
wood rabbit, cottontail, cottontail rabbit (score = 0.04976)
```

#### 补充：迁移学习

（1）

迁移学习（Transfer learning）顾名思义就是就是把已学训练好的模型参数迁移到新的模型来帮助新模型训练数据集。*——from：https://feisky.xyz/machine-learning/transfer-learning.html*

（2）

深度学习可以说是一门数据驱动的学科，各种有名的 CNN 模型，无一不是在大型的数据库上进行的训练。像 ImageNet 这种规模的数据库，动辄上百万张图片。对于普通的机器学习工作者、学习者来说，面对的任务各不相同，很难拿到如此大规模的数据集。同时也没有谷歌，Facebook 那种大公司惊人的算力支持，想从 0 训练一个深度 CNN 网络，基本是不可能的。但是好在已经训练好的模型的参数，往往经过简单的调整和训练，就可以很好的迁移到其他不同的数据集上，同时也无需大量的算力支撑，便能在短时间内训练得出满意的效果。这便是迁移学习。究其根本，就是虽然图像的数据集不同，但是底层的特征却是有大部分通用的。

**迁移学习主要分为两种：**

- 第一种即所谓的 transfer learning，迁移训练时，移掉最顶层，比如 ImageNet 训练任务的顶层就是一个 1000 输出的全连接层，换上新的顶层，比如输出为 10 的全连接层，然后训练的时候，只训练最后两层，即原网络的倒数第二层和新换的全连接输出层。可以说 transfer learning 将底层的网络当做了一个特征提取器来使用。
- 第二种叫做 fine tune，和 transfer learning一样，换一个新的顶层，但是这一次在训练的过程中，所有的（或大部分）其它层都会经过训练。也就是底层的权重也会随着训练进行调整。

一个典型的迁移学习过程是这样的。首先通过 transfer learning 对新的数据集进行训练，训练过一定 epoch 之后，改用 fine tune 方法继续训练，同时降低学习率。这样做是因为如果一开始就采用 fine tune 方法的话，网络还没有适应新的数据，那么在进行参数更新的时候，比较大的梯度可能会导致原本训练的比较好的参数被污染，反而导致效果下降。

> *——from：[使用 Google Inception V3模型进行迁移学习之——牛津大学花朵分类 | Yong's Blog](<https://imyong.top/2018/05/30/Inception-V3-implementation-17-flowers-classes/#1.%E6%A6%82%E8%BF%B0>)*

（3）

另外在一篇硕士论文也有提到迁移学习，引用如下：

> 为了解决神经网络训练需要标注大量数据，并且网络优化困难，收敛速度慢的问题，本文采用了迁移学习的方法训练设计好的神经网络。
>
> ​	第一步，在收集到的网络公共数据集上进行预训练，通过 Camvid、Cityscape、KITTI 等公共数据集对设计的卷积网络进行初始训练，得到初始模型。
>
> ​	第二步，在仿真数据集上，使用第一步训练好的初始模型初始化网络参数，进行再训练，得到针对道路场景的初始分割模型。
>
> ​	第三步，在手工采集并标注的道路数据集上，使用第二步训练得到的分割模型初始化网络权重，进行最终的训练，得到最终的高精度模型。
>
> 通过迁移学习的方法可以有效提高在有限训练样本情况下的模型精度，提高模型训练速度。

相关阅读：

- [什么是迁移学习 (Transfer Learning)？这个领域历史发展前景如何？ - 知乎](<https://www.zhihu.com/question/41979241>)