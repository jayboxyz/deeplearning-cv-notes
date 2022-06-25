## TensorFlow进行结构可视化

### 一、演示如何让Tensorboard进行结构可视化：

（对应代码：`5-2tensorboard网络结构.py`）

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

# 命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('layer'):
    # 创建一个简单的神经网络
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
```

通过使用 `with tf.name_scope('input')` 来设置命名空间标记可视化参数，程序运行之后将在当前目录生成一个 logs 目录，目录下有如下内容：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-92630115.jpg)

然后 cmd 下运行下面这个命令：`tensorboard --logdir=D:\Tensorflow\logs`

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-3161246.jpg)

打开浏览器，输入`http://Jaybo-pc:6006`，可以看到：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-11610986.jpg)

然后可以点击观察里面的一些细节：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-4454287.jpg)

### 二、演示如何让参数细节可视化，绘制各个参数变化情况

#### 参数细节及变化可视化

（对应代码：`5-3tensorboard网络运行.py`）

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
```



``` python
# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图
```



``` python
# 命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('layer'):
    # 创建一个简单的神经网络
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(51):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})

        writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
```

Tensorboard 内容大致如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-53721827.jpg)

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-15066889.jpg)

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-54883213.jpg)

#### 个人补充：TensorBoard可视化

关于使用可视化工具 TensorBoard 的更多的学习和实践：[Tensorflow的可视化工具Tensorboard的初步使用](https://blog.csdn.net/sinat_33761963/article/details/62433234)

TensorBoard 可以记录与展示以下数据形式： 

（1）标量 Scalars 

（2）图片 Images 

（3）音频 Audio 

（4）计算图 Graph 

（5）数据分布 Distribution 

（6）直方图 Histograms 

（7）嵌入向量 Embeddings

Tensorboard 的可视化过程：

（1）首先肯定是先建立一个 graph，你想从这个 graph 中获取某些数据的信息

（2）确定要在 graph 中的哪些节点放置 summary operations 以记录信息 

- 使用 `tf.summary.scalar` 记录标量 
- 使用 `tf.summary.histogram` 记录数据的直方图 
- 使用 `tf.summary.distribution` 记录数据的分布图 
- 使用 `tf.summary.image` 记录图像数据 
- ….

（3）operations 并不会去真的执行计算，除非你告诉他们需要去 run，或者它被其他的需要 run 的 operation 所依赖。而我们上一步创建的这些 summary operations 其实并不被其他节点依赖，因此，我们需要特地去运行所有的 summary 节点。但是呢，一份程序下来可能有超多这样的 summary 节点，要手动一个一个去启动自然是及其繁琐的，因此我们可以使用 `tf.summary.merge_all` 去将所有 summary 节点合并成一个节点，只要运行这个节点，就能产生所有我们之前设置的 summary data。

（4）使用`tf.summary.FileWriter`将运行后输出的数据都保存到本地磁盘中

（5）运行整个程序，并在命令行输入运行 tensorboard 的指令，之后打开 web 端可查看可视化的结果。

再看下该文：[tensorboard快速上手，tensorboard可视化普及贴（代码基于tensorflow1.2以上）](http://nooverfit.com/wp/tensorboard%E4%B8%8A%E6%89%8B%EF%BC%8Ctensorboard%E5%8F%AF%E8%A7%86%E5%8C%96%E6%99%AE%E5%8F%8A%E8%B4%B4%EF%BC%88%E4%BB%A3%E7%A0%81%E5%9F%BA%E4%BA%8Etensorflow1-2%E4%BB%A5%E4%B8%8A%EF%BC%89/)

总的来说就是除了可视化模型的 Graph，如果我们需要流图训练过程中动态日志 log，比如现在还没有动态scalars（标量值）数据，所以我们可以定义一些 log summary 的操作（下面是对 cost 和 accuracy 标量打 log）：

``` python
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
```

定义完成后，我们不需要逐条执行上述操作，只需用 merge 操作一并执行：

``` python
summary_op = tf.summary.merge_all()
```

最后在流图真正流动训练的时候，记得执行，并写入上述操作到 log 中：

``` python
_, summary = sess.run([train_op, summary_op], feed_dict={x: batch_x, y_: batch_y})
            
# write log
writer.add_summary(summary, epoch * batch_count+i)
```

其中，`add_summary()`方法的 ***第二个参数是 scalar 图标坐标中的 x 轴的值***，summary 对象 ***计算出的标量是 y 轴的值***，如图：

![](http://nooverfit.com/wp/wp-content/uploads/2018/03/QQ%E6%88%AA%E5%9B%BE20180319163020.png)

关于图中 Smoothing 顺带提下，它的大小是指什么意思呢？参看该文：[tensorboard 界面smooth参数实现](https://dingguanglei.com/tensorboard-xia-smoothgong-neng-tan-jiu/)

其实就是指的作图时曲线的平滑程度。调整 Smoothing 参数，控制曲线平滑处理，数值越小越接近实际值，波动大；数值越大曲线越平缓。如果不平滑处理的话，有些曲线波动很大，难以看出趋势。0 就是不平滑处理，1 就是最平滑。例如：

当`smooth = 0` 时：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181229213044.png)

当`smooth = 0.5` 时：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181229213112.png)

当`smooth = 0.9` 时：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181229213134.png)

#### 手写数字识别Embeding

下面进行手写数字识别 Embeding（[官网链接](https://www.tensorflow.org/guide/embedding)）可视化过程：（对应代码：`5-4tensorboard可视化.py`）

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-21761405.jpg)

先把 Embeding 文件`mnist_10k_sprite.png`粘贴到如上 data 文件夹下。

``` python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

# 载入mnist数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 运行次数
max_steps = 1001
# 图片数量
image_num = 3000
# 文件路径 
#DIR = "D:\\TensorFlow" # 路径这样写也可以
DIR = "D:/Tensorflow/"

# 定义会话
sess = tf.Session()

# 载入图片
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图
```



``` python
# 命名空间
with tf.name_scope('input'):
    # 这里的none表示第一个维度可以是任意的长度
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # 正确的标签
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 显示图片
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

with tf.name_scope('layer'):
    # 创建一个简单神经网络
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    # 交叉熵代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
sess.run(tf.global_variables_initializer())

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 把correct_prediction变为float32类型
        tf.summary.scalar('accuracy', accuracy)

# 产生metadata文件
if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

        # 合并所有的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer, config)

for i in range(max_steps):
    # 每个批次100个样本
    batch_xs, batch_ys = mnist.train.next_batch(100)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options,
                          run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)

    if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(i) + ", Testing Accuracy= " + str(acc))

saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
projector_writer.close()
sess.close()
```

运行结果：

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0, Testing Accuracy= 0.2994
Iter 100, Testing Accuracy= 0.8024
Iter 200, Testing Accuracy= 0.8202
Iter 300, Testing Accuracy= 0.8305
Iter 400, Testing Accuracy= 0.8321
Iter 500, Testing Accuracy= 0.8688
Iter 600, Testing Accuracy= 0.8893
Iter 700, Testing Accuracy= 0.8994
Iter 800, Testing Accuracy= 0.9012
Iter 900, Testing Accuracy= 0.904
Iter 1000, Testing Accuracy= 0.9054
```

程序运行完毕，最后会在`D:\TensorFlow\projector\projector`文件夹下生成如下文件：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-9243596.jpg)

然后 cmd 下运行：`tensorboard --logdir=D:\TensorFlow\projector\projector`

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-75085803.jpg)



