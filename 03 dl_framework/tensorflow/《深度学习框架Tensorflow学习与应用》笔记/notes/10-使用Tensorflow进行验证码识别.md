## 使用TensorFlow进行验证码识别

### 一、关于tfrecord文件

于 Tensorflow 读取数据，官网给出了三种方法：

- 供给数据(Feeding)：在 TensorFlow 程序运行的每一步， 让 Python 代码来供给数据。

- 从文件读取数据：在 TensorFlow 图的起始， 让一个输入管线从文件中读取数据。

- 预加载数据：在 TensorFlow 图中定义常量或变量来保存所有数据（仅适用于数据量比较小的情况）。

对于数据量较小而言，可能一般选择直接将数据加载进内存，然后再分 batch 输入网络进行训练（tip：使用这种方法时，结合 yield 使用更为简洁）。但是，如果数据量较大，这样的方法就不适用了，因为太耗内存，所以这时最好使用 tensorflow 提供的队列 queue，也就是第二种方法 从文件读取数据，使用 tensorflow 内定标准格式——TFRecords。

从宏观来讲，tfrecord 其实是一种数据存储形式。使用 tfrecord 时，实际上是先读取原生数据，然后转换成 tfrecord 格式，再存储在硬盘上。而使用时，再把数据从相应的 tfrecord 文件中解码读取出来。那么使用 tfrecord 和直接从硬盘读取原生数据相比到底有什么优势呢？其实，Tensorflow 有和 tfrecord 配套的一些函数，可以加快数据的处理。实际读取 tfrecord 数据时，先以相应的 tfrecord 文件为参数，创建一个输入队列，这个队列有一定的容量（视具体硬件限制，用户可以设置不同的值），在一部分数据出队列时，tfrecord 中的其他数据就可以通过预取进入队列，并且这个过程和网络的计算是独立进行的。也就是说，网络每一个 iteration 的训练不必等待数据队列准备好再开始，队列中的数据始终是充足的，而往队列中填充数据时，也可以使用多线程加速。

相关资料：

- [tensorflow读取数据-tfrecord格式](https://blog.csdn.net/happyhorizion/article/details/77894055)
- [Tensorflow中使用tfrecord方式读取数据](https://blog.csdn.net/u010358677/article/details/70544241)
- [Tensorflow 训练自己的数据集（二）（TFRecord）](https://blog.csdn.net/Best_Coder/article/details/70146441)

### 二、TensorFlow进行验证码识别

#### 1、两种验证方法

1）验证码识别方法一

- 把标签转为向量，向量长度为 40。比如有一个验证码为 0782 ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-45205058.jpg)

- 它的标签可以转为长度为 40 的向量：1000000000 0000000100 0000000010 0010000000

- 训练方法跟 0-9 手写数字识别类似。

2）验证码识别方法二

- 拆分为 4 个标签，比如有一个验证码为 0782

- 四个标签：

  ``` xml
  Label0：1000000000
  Label1：0000000100
  Label2：0000000010
  Label3：0010000000
  ```

- 可以使用多任务学习

#### 2、多任务学习

多个相似任务有多个数据集，交替训练：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-89492107.jpg)

多个相似任务只有一个数据集，联合训练：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-12109228.jpg)

#### 3、采用多任务联合训练进行验证码识别

第一种方法相对来说简单，这节介绍使用第二种方法：对验证码拆分为 4 个标签，比如有一个验证码为 0782，4 个标签分别对应：

``` xml
Label0：1000000000
Label1：0000000100
Label2：0000000010
Label3：0010000000
```

##### 验证码生成

（1）生成 10000 张验证码图片

先安装： `pip install captcha`。

完整代码如下：（对应代码：`10-1验证码生成.py`）

``` python
# coding: utf-8

# 验证码生成库
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random
import sys
 
number = ['0','1','2','3','4','5','6','7','8','9']
# alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def random_captcha_text(char_set=number, captcha_size=4):
    # 验证码列表
    captcha_text = []
    for i in range(captcha_size):
        #随机选择
        c = random.choice(char_set)
        #加入验证码列表
        captcha_text.append(c)
    return captcha_text
 
# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    #获得随机生成的验证码
    captcha_text = random_captcha_text()
    #把验证码列表转为字符串
    captcha_text = ''.join(captcha_text)
    #生成验证码
    captcha = image.generate(captcha_text)
    image.write(captcha_text, 'captcha/images/' + captcha_text + '.jpg')  # 写到文件

#数量少于10000，因为重名
num = 10000
if __name__ == '__main__':
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>> Creating image %d/%d' % (i+1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
                        
    print("生成完毕")
```

运行结果：

``` xml
>> Creating image 10000/10000
生成完毕
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-43920597.jpg)

每张验证码图片像素大小为`160*60`：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-56189029.jpg)



##### 生成tfrecord文件

（2）生成 tfrecord 文件

把验证码转换为 tfrecord 文件，这次不用切分，只转换为一个块。

完整代码如下：（对应代码：`10-2生成tfrecord文件.py`）

``` python
# coding: utf-8

import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np

#验证集数量
_NUM_TEST = 500

#随机种子
_RANDOM_SEED = 0

#数据集路径
DATASET_DIR = "D:/Tensorflow/captcha/images/"

#tfrecord文件存放路径
TFRECORD_DIR = "D:/Tensorflow/captcha/"


#判断tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        output_filename = os.path.join(dataset_dir,split_name + '.tfrecords')
        if not tf.gfile.Exists(output_filename):
            return False
    return True

#获取所有验证码图片
def _get_filenames_and_classes(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        #获取文件路径
        path = os.path.join(dataset_dir, filename)
        photo_filenames.append(path)
    return photo_filenames

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, label0, label1, label2, label3):
    #Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
      'image': bytes_feature(image_data),
      'label0': int64_feature(label0),
      'label1': int64_feature(label1),
      'label2': int64_feature(label2),
      'label3': int64_feature(label3),
    }))

#把数据转为TFRecord格式
def _convert_dataset(split_name, filenames, dataset_dir):
    assert split_name in ['train', 'test']

    with tf.Session() as sess:
        #定义tfrecord文件的路径+名字
        output_filename = os.path.join(TFRECORD_DIR,split_name + '.tfrecords')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,filename in enumerate(filenames):
                try:
                    sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                    sys.stdout.flush()

                    #读取图片
                    image_data = Image.open(filename)  
                    #根据模型的结构resize
                    image_data = image_data.resize((224, 224))
                    #灰度化
                    image_data = np.array(image_data.convert('L'))
                    #将图片转化为bytes
                    image_data = image_data.tobytes()              

                    #获取label
                    labels = filename.split('/')[-1][0:4]
                    num_labels = []
                    for j in range(4):
                        num_labels.append(int(labels[j]))
                                            
                    #生成protocol数据类型
                    example = image_to_tfexample(image_data, num_labels[0], num_labels[1], num_labels[2], num_labels[3])
                    tfrecord_writer.write(example.SerializeToString())
                    
                except IOError as e:
                    print('Could not read:',filename)
                    print('Error:',e)
                    print('Skip it\n')
    sys.stdout.write('\n')
    sys.stdout.flush()

#判断tfrecord文件是否存在
if _dataset_exists(TFRECORD_DIR):
    print('tfcecord文件已存在')
else:
    #获得所有图片
    photo_filenames = _get_filenames_and_classes(DATASET_DIR)

    #把数据切分为训练集和测试集,并打乱
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[_NUM_TEST:]
    testing_filenames = photo_filenames[:_NUM_TEST]

    #数据转换
    _convert_dataset('train', training_filenames, DATASET_DIR)
    _convert_dataset('test', testing_filenames, DATASET_DIR)
    print('生成tfcecord文件')
```

最后生成：（截图自视频）

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012142033.png)

##### 验证码识别

（3）验证码识别

拷贝 `slim\nets\` 目录到我们程序代码的相同路径（在下面的验证码识别代码中我们会导入 nets）。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012183851.png)

然后修改 `nets\alexnet.py` 网络结构以满足我们的需求，修改后的文件内容：

``` python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def alexnet_v2_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2'):

  with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
      net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      net = slim.conv2d(net, 192, [5, 5], scope='conv2')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = slim.conv2d(net, 384, [3, 3], scope='conv3')
      net = slim.conv2d(net, 384, [3, 3], scope='conv4')
      net = slim.conv2d(net, 256, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

      # Use conv2d instead of fully_connected layers.
      with slim.arg_scope([slim.conv2d],
                          weights_initializer=trunc_normal(0.005),
                          biases_initializer=tf.constant_initializer(0.1)):
        net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                          scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net0 = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          biases_initializer=tf.zeros_initializer(),
                          scope='fc8_0')
        net1 = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          biases_initializer=tf.zeros_initializer(),
                          scope='fc8_1')
        net2 = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          biases_initializer=tf.zeros_initializer(),
                          scope='fc8_2')
        net3 = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          biases_initializer=tf.zeros_initializer(),
                          scope='fc8_3')

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net0 = tf.squeeze(net0, [1, 2], name='fc8_0/squeezed')
        end_points[sc.name + '/fc8_0'] = net0
        net1 = tf.squeeze(net1, [1, 2], name='fc8_1/squeezed')
        end_points[sc.name + '/fc8_1'] = net1
        net2 = tf.squeeze(net2, [1, 2], name='fc8_2/squeezed')
        end_points[sc.name + '/fc8_2'] = net2
        net3 = tf.squeeze(net3, [1, 2], name='fc8_3/squeezed')
        end_points[sc.name + '/fc8_3'] = net3


      return net0,net1,net2,net3,end_points
alexnet_v2.default_image_size = 224
```

完整代码如下：（对应代码：`10-3验证码识别.py`）

``` python
# coding: utf-8

import os
import tensorflow as tf 
from PIL import Image
from nets import nets_factory
import numpy as np

# 不同字符数量
CHAR_SET_LEN = 10
# 图片高度
IMAGE_HEIGHT = 60 
# 图片宽度
IMAGE_WIDTH = 160  
# 批次
BATCH_SIZE = 25
# tfrecord文件存放路径
TFRECORD_FILE = "D:/Tensorflow/captcha/train.tfrecords"

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])  
y0 = tf.placeholder(tf.float32, [None]) 
y1 = tf.placeholder(tf.float32, [None]) 
y2 = tf.placeholder(tf.float32, [None]) 
y3 = tf.placeholder(tf.float32, [None])

# 学习率
lr = tf.Variable(0.003, dtype=tf.float32)

# 从tfrecord读出数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image' : tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    # 获取图片数据
    image = tf.decode_raw(features['image'], tf.uint8)
    # tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image, [224, 224])
    # 图片预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # 获取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, label0, label1, label2, label3
```



``` python
# 获取图片数据和标签
image, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

#使用shuffle_batch可以随机打乱
image_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, label0, label1, label2, label3], batch_size = BATCH_SIZE,
        capacity = 50000, min_after_dequeue=10000, num_threads=1)

#定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=True)
 
    
with tf.Session() as sess:
    # inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    # 数据输入网络得到输出值
    logits0,logits1,logits2,logits3,end_points = train_network_fn(X)
    
    # 把标签转成one_hot的形式
    one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_LEN)
    
    # 计算loss
    loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits0,labels=one_hot_labels0)) 
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1,labels=one_hot_labels1)) 
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2,labels=one_hot_labels2)) 
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits3,labels=one_hot_labels3)) 
    # 计算总的loss
    total_loss = (loss0+loss1+loss2+loss3)/4.0
    # 优化total_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss) 
    
    # 计算准确率
    correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0,1),tf.argmax(logits0,1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0,tf.float32))
    
    correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1,1),tf.argmax(logits1,1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1,tf.float32))
    
    correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2,1),tf.argmax(logits2,1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2,tf.float32))
    
    correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3,1),tf.argmax(logits3,1))
    accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3,tf.float32)) 
    
    # 用于保存模型
    saver = tf.train.Saver()
    # 初始化
    sess.run(tf.global_variables_initializer())
    
    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(6001):
        # 获取一个批次的数据和标签
        b_image, b_label0, b_label1 ,b_label2 ,b_label3 = sess.run([image_batch, label_batch0, label_batch1, label_batch2, label_batch3])
        # 优化模型
        sess.run(optimizer, feed_dict={x: b_image, y0:b_label0, y1: b_label1, y2: b_label2, y3: b_label3})  

        # 每迭代20次计算一次loss和准确率  
        if i % 20 == 0:  
            # 每迭代2000次降低一次学习率
            if i%2000 == 0:
                sess.run(tf.assign(lr, lr/3))
            acc0,acc1,acc2,acc3,loss_ = sess.run([accuracy0,accuracy1,accuracy2,accuracy3,total_loss],feed_dict={x: b_image,
                                                                                                                y0: b_label0,
                                                                                                                y1: b_label1,
                                                                                                                y2: b_label2,
                                                                                                                y3: b_label3})      
            learning_rate = sess.run(lr)
            print ("Iter:%d  Loss:%.3f  Accuracy:%.2f,%.2f,%.2f,%.2f  Learning_rate:%.4f" % (i,loss_,acc0,acc1,acc2,acc3,learning_rate))
             
            # 保存模型
            # if acc0 > 0.90 and acc1 > 0.90 and acc2 > 0.90 and acc3 > 0.90: 
            if i==6000:
                saver.save(sess, "./captcha/models/crack_captcha.model", global_step=i)  
                break 
                
    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)
```

运行结果：

``` xml
Iter:0  Loss:1713.653  Accuracy:0.24,0.24,0.16,0.32  Learning_rate:0.0010
Iter:20  Loss:2.303  Accuracy:0.16,0.00,0.08,0.08  Learning_rate:0.0010
Iter:40  Loss:2.308  Accuracy:0.16,0.04,0.08,0.12  Learning_rate:0.0010
Iter:60  Loss:2.296  Accuracy:0.16,0.04,0.28,0.08  Learning_rate:0.0010
Iter:80  Loss:2.298  Accuracy:0.20,0.00,0.20,0.08  Learning_rate:0.0010
Iter:100  Loss:2.303  Accuracy:0.16,0.08,0.04,0.08  Learning_rate:0.0010
Iter:120  Loss:2.297  Accuracy:0.12,0.08,0.08,0.04  Learning_rate:0.0010
Iter:140  Loss:2.300  Accuracy:0.08,0.12,0.24,0.16  Learning_rate:0.0010
Iter:160  Loss:2.297  Accuracy:0.12,0.08,0.16,0.20  Learning_rate:0.0010
Iter:180  Loss:2.305  Accuracy:0.08,0.08,0.08,0.16  Learning_rate:0.0010
Iter:200  Loss:2.301  Accuracy:0.20,0.04,0.16,0.12  Learning_rate:0.0010
Iter:220  Loss:2.295  Accuracy:0.04,0.12,0.04,0.12  Learning_rate:0.0010
Iter:240  Loss:2.311  Accuracy:0.16,0.16,0.08,0.08  Learning_rate:0.0010
Iter:260  Loss:2.297  Accuracy:0.08,0.00,0.20,0.08  Learning_rate:0.0010
Iter:280  Loss:2.298  Accuracy:0.20,0.12,0.12,0.16  Learning_rate:0.0010
Iter:300  Loss:2.307  Accuracy:0.16,0.16,0.12,0.00  Learning_rate:0.0010
Iter:320  Loss:2.276  Accuracy:0.04,0.04,0.20,0.20  Learning_rate:0.0010
Iter:340  Loss:2.276  Accuracy:0.12,0.12,0.16,0.16  Learning_rate:0.0010
Iter:360  Loss:2.301  Accuracy:0.16,0.24,0.24,0.08  Learning_rate:0.0010
Iter:380  Loss:2.275  Accuracy:0.08,0.12,0.20,0.20  Learning_rate:0.0010
Iter:400  Loss:2.184  Accuracy:0.28,0.12,0.12,0.16  Learning_rate:0.0010
Iter:420  Loss:2.168  Accuracy:0.28,0.16,0.20,0.20  Learning_rate:0.0010
Iter:440  Loss:2.221  Accuracy:0.00,0.12,0.24,0.24  Learning_rate:0.0010
Iter:460  Loss:2.197  Accuracy:0.08,0.24,0.16,0.20  Learning_rate:0.0010
Iter:480  Loss:2.106  Accuracy:0.24,0.12,0.12,0.20  Learning_rate:0.0010
Iter:500  Loss:2.022  Accuracy:0.32,0.20,0.16,0.28  Learning_rate:0.0010
......(省略)
```

##### 验证码测试

（4）验证码测试

完整代码如下：（对应代码：`10-4captcha_test.py`）

``` python
# coding: utf-8

import os
import tensorflow as tf 
from PIL import Image
from nets import nets_factory
import numpy as np
import matplotlib.pyplot as plt  

# 不同字符数量
CHAR_SET_LEN = 10
# 图片高度
IMAGE_HEIGHT = 60 
# 图片宽度
IMAGE_WIDTH = 160  
# 批次
BATCH_SIZE = 1
# tfrecord文件存放路径
TFRECORD_FILE = "D:/Tensorflow/captcha/test.tfrecords"

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])  

# 从tfrecord读出数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image' : tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    # 获取图片数据
    image = tf.decode_raw(features['image'], tf.uint8)
    # 没有经过预处理的灰度图
    image_raw = tf.reshape(image, [224, 224])
    # tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image, [224, 224])
    # 图片预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # 获取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, image_raw, label0, label1, label2, label3
```



``` python
# 获取图片数据和标签
image, image_raw, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

#使用shuffle_batch可以随机打乱
image_batch, image_raw_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, image_raw, label0, label1, label2, label3], batch_size = BATCH_SIZE,
        capacity = 50000, min_after_dequeue=10000, num_threads=1)

#定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=False)

with tf.Session() as sess:
    # inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    # 数据输入网络得到输出值
    logits0,logits1,logits2,logits3,end_points = train_network_fn(X)
    
    # 预测值
    predict0 = tf.reshape(logits0, [-1, CHAR_SET_LEN])  
    predict0 = tf.argmax(predict0, 1)  

    predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])  
    predict1 = tf.argmax(predict1, 1)  

    predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])  
    predict2 = tf.argmax(predict2, 1)  

    predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])  
    predict3 = tf.argmax(predict3, 1)  

    # 初始化
    sess.run(tf.global_variables_initializer())
    # 载入训练好的模型
    saver = tf.train.Saver()
    saver.restore(sess,'./captcha/models/crack_captcha.model-6000')

    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        # 获取一个批次的数据和标签
        b_image, b_image_raw, b_label0, b_label1 ,b_label2 ,b_label3 = sess.run([image_batch, 
                                                                    image_raw_batch, 
                                                                    label_batch0, 
                                                                    label_batch1, 
                                                                    label_batch2, 
                                                                    label_batch3])
        # 显示图片
        img=Image.fromarray(b_image_raw[0],'L')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        # 打印标签
        print('label:',b_label0, b_label1 ,b_label2 ,b_label3)
        # 预测
        label0,label1,label2,label3 = sess.run([predict0,predict1,predict2,predict3], feed_dict={x: b_image})
        # 打印预测值
        print('predict:',label0,label1,label2,label3) 
                
    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)
```

测试结果：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012142132.png)

``` xml
label: [5] [1] [3] [7]
predict: [5] [0] [3] [7]
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012142044.png)

``` xml
label: [6] [3] [5] [0]
predict: [6] [3] [5] [0]
```

......

补充内容：假如识别的验证码不仅有 0\~9 的数字，还有 A\~Z 这样 26 个英文字母，怎么办？可以延长向量的表示（长度 36）：

``` xml
表示1：1000000000 00000000000000000000000000
表示2：0100000000 00000000000000000000000000
...
表示A：0000000000 10000000000000000000000000
表示B：0000000000 01000000000000000000000000
...
表示Z：0000000000 00000000000000000000000001
```

