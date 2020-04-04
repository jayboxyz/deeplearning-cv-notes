## TensorFlow的GPU版本安装，设计并训练自己的模型进行图像识别

### 一、TensorFlow的GPU版本安装

#### 1、Windows平台

举例：win7 64位环境下安装 TensorFlow 的 GPU 版本。

1）安装 NVIDIA 显卡驱动程序

下载地址：https://www.nvidia.cn/Download/index.aspx?lang=cn

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-26477203.jpg)

选择适合自己电脑的显卡驱动下载。安装过程很简单，直接下一步就可以。

2）安装 CUNA

> CUDA（Compute Unified Device Architecture）是英伟达公司推出的一种基于新的并行编程模型和指令集架构的通用计算架构，它能利用英伟达 GPU 的并行计算引擎，比 CPU 更高效的解决许多复杂计算任务。
>
> 简单地说，CUDA 是 NVIDIA 推出的用于自家 GPU 的并行计算框架，也就是说 CUDA 只能在 NVIDIA 的 GPU 上运行，而且只有当要解决的计算问题是可以大量并行计算的时候才能发挥CUDA的作用。
>
> 关于 CUDA 和 cuDNN 的认识：[GPU，CUDA，cuDNN的理解](https://blog.csdn.net/u014380165/article/details/77340765)

下载 CUDA 前，先确认 GPU 显卡所支持的 CUDA 版本：

> 控制面板 --> NVIDIA控制面板 --> 帮助 --> 系统信息 --> 组件 --> nvidia.dll 后面的 cuda 参数
>
> ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-83310095.jpg)

下载对应版本的 CUDA，下载地址：https://developer.nvidia.com/cuda-downloads

下载完成，安装也很简单直接下一步就好。

安装好之后把 CUDA 安装目录下的 bin 和 lib\x64 添加到 Path 环境变量中。

3）安装 cuDNN

> cuDNN（CUDA Deep Neural Network library）：是 NVIDIA 打造的针对深度神经网络的加速库，是一个用于深层神经网络的 GPU 加速库。如果你要用 GPU 训练模型，cuDNN 不是必须的，但是一般会采用这个加速库。

注意： GPU 显卡计算能力大于 3.0 才支持 cuDNN。详情：https://developer.nvidia.com/cuda-gpus

cuDNN下载：https://developer.nvidia.com/rdp/cudnn-download，需要注册才能下载。

下载完成之后再：

``` xml
a）解压压缩包，把压缩包中 bin、include、lib 中的文件分别拷贝到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0 目录下对应目录中

b）把C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\CUPTI\libx64\cupti64_80.dll拷贝到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin
```

到此 cuDNN 安装完成。

4）安装 tensorflow-gpu

``` xml
pip uninstall tensorflow	#先卸载安装过的cpu版tensorflow
pip install tensorflow-gpu  #安装gpu版的tensorflow
```

PS：只要安装了 GPU 版的 TensorFlow，在计算的时候会自动调用 GPU 来进行运算。

#### 2、Linux平台

- [ubuntu 16.04 安装Tensorflow(CPU和GPU)](https://blog.csdn.net/jiang_z_q/article/details/73264561)

### 二、使用inception-v3模型进行训练和测试

先了解下：

- [使用自己的数据集训练GoogLenet InceptionNet V1 V2 V3模型（TensorFlow）](https://blog.csdn.net/guyuealian/article/details/81560537)
- ......

#### 1、介绍

训练自己的网络模型，有三个办法：

1. 从无到有，自己从头构建网络，从头训练；

   > 一开始参数都是随机的，把我们准备好的数据一个一个批次的放进去训练，经过非常长的时间把模型训练好。采用该方式，我们可能需要准备大量的数据集。

2. 用一个现成的质量比较好的模型，固定前面参数，在后面添加几层，训练后面的参数；

3. 改造现成的质量比较好的模型，训练整个网络的模型（初始层的学习率比较低）。

比如使用第二种办法，对已经训练好了 inception-v3 模型进行改造，因为是已经训练好的，所以里面的权值参数是已经确定的。

如下图，inception-v3 模型的 pool3 位置下面的所有卷积层、池化层的参数都固定，它们是大量图片训练出来的参数，如果你也是做图像分类的话，底下那些参数都是非常好的参数，我们直接拿来用（毕竟底下的卷积都是用来做图像特征提取的，而这些特征都是经过大量图片数据学习到的），也适用我们的任务。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-92827217.jpg)

图中 softmax 是最后用于做分类的，1000 个分类，注意到右侧多出了一路 input，多出的这一路是用来对自己的图片分类任务新添加的结构。假如我们要对图像 5 分类，pool3 连接的 input 这一路，就可以用来进行 5 分类，这样，前面的所有层都不用训练，只训练最后一层，最后一层就是用来做分类的。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-54741870.jpg)



#### 2、模型训练

先在磁盘下准备这样一个目录，比如在 D 盘：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-4566670.jpg)

其中 bottleneck 用来保存训练过程的中间文件，data 文件夹里面内容如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-32645120.jpg)

这里需要图片分类，一共五个类，每个类对应于一个文件夹，每个文件夹里面为该类的训练文件，这里以 flower 为例：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-18908364.jpg)

每个文件夹里有 500 张图片。图片数据集可以从该网站下载：http://www.robots.ox.ac.uk/~vgg/data/

下面是我下载的数据集地址：

- animal：http://www.robots.ox.ac.uk/~vgg/data/pets/	（`images.tar.gz`，~765M）
- flower：http://www.robots.ox.ac.uk/~vgg/data/flowers/  （`17flowers.tgz`，~58.8M）
- plane：http://www.robots.ox.ac.uk/~vgg/data/airplanes_side/airplanes_side.tar  （`airplanes_side.tar`，~43.7M）
- house：http://www.robots.ox.ac.uk/~vgg/data/houses/houses.tar  （`houses.tar`，~16.9M）
- guitar：http://www.robots.ox.ac.uk/~vgg/data/guitars/guitars.tar  （`guitars.tar`，~24.5M）

从每个数据集选择 500 张自己想要的图片粘贴到 data 目录下对应的文件夹下。

接下来是 images 文件夹，里面保存用来识别测试的图片，这里给出了五张图片，对应于五类需要分类的图片：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-61891662.jpg)

inception_model 文件夹里是 inception-v3 模型，之后的训练将要用到这个模型的结构：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-47071127.jpg)

上面的工作做完之后，我们可以使用官方提供的程序进行训练了。

首先下载 tensorflow 的源码，GitHub 地址：https://github.com/tensorflow/tensorflow，解压并放在指定位置，比如 `D:\TensorFlow`目录下。然后写个批处理文件去执行 TensorFlow 中`retrain.py`程序，自动训练模型。

> !!!需要注意下：在这里选择 tensorflow 1.4 版本下载，因为新版 tensorflow 源码已将该部分内容转移，能看到新版 tensorflow 的 image_retaining 文件夹下有个 README.md 文件中已经提到了，内容粘贴如下：
>
> ``` xml
> NOTE: This code has moved to
> https://github.com/tensorflow/hub/tree/master/examples/image_retraining
> 
> retrain.py is an example script that shows how one can adapt a pretrained
> network for other classification problems (including use with TFLite and
> quantization).
> 
> As of TensorFlow 1.7, it is recommended to use a pretrained network from
> TensorFlow Hub, using the new version of this example found in the location
> above, as explained in TensorFlow's revised image retraining
> tutorial.
> 
> Older versions of this example (using frozen GraphDefs instead of
> TensorFlow Hub modules) are available in the release branches of
> TensorFlow versions up to and including 1.7.
> ```

在`D:\retrain\`文件夹下新建`retrain.bat` ，其内容如下：

``` xml
python D:/Tensorflow/tensorflow-1.4.0/tensorflow/examples/image_retraining/retrain.py ^
--bottleneck_dir bottleneck ^
--how_many_training_steps 200 ^
--model_dir inception_model/ ^
--output_graph output_graph.pb ^
--output_labels output_labels.txt ^
--image_dir data/train/
pause
```

解释下：

- 第一行表示：使用 tensorflow 中的 `retrain.py`程序
- `^`：连接符，起到连接的作用
- bottleneck_dir：每张输入图片在 pool3 会得到一个数据，保存到 bottleneck_dir 指定的目录下
- how_many_training_steps：训练的周期
- model_dir：指定使用的模型，比如该案例使用的为 inception-v3 模型，指定到该模型的目录，会自动找`inception-2015-12-05.tgz`文件
- output_graph：输出一个训练后的模型，指定输出位置
- output_labels：输出标签
- image_dir：要分类训练的图片

最后，执行脚本开始训练，训练结束后，我们得到两个文件：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-43976984.jpg)

其中 output\_graph.pb 为训练后得到的模型，output\_labels.txt 文件内容如下：

``` xml
animal
flower
guitar
house
plane
```

为训练集合的标签分类。

注意：output\_labels.txt  文件用 Windows7 自带的文本编辑器打开显示的是标签分类是连在一起，没有换行，官方是有换行的，用 Notepad++ 编辑器是如上正常显示。

#### 3、模型测试

完整代码如下：（对应代码：`9-1测试训练好的模型.py`）

``` python
# coding: utf-8

import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

lines = tf.gfile.GFile('retain/output_labels.txt').readlines()
uid_to_human = {}
# 一行一行读取数据
for uid, line in enumerate(lines):
    # 去掉换行符
    line=line.strip('\n')
    uid_to_human[uid] = line

def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]
```



``` python
# 创建一个图来存放google训练好的模型
with tf.gfile.FastGFile('retain/output_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
```



``` python
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #遍历目录
    for root,dirs,files in os.walk('retain/images/'):	#指定测试图片的位置
        for file in files:
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root,file), 'rb').read()
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})#图片格式是jpg格式
            predictions = np.squeeze(predictions)#把结果转为1维数据

            #打印图片路径及名称
            image_path = os.path.join(root,file)
            print(image_path)

            #排序
            top_k = predictions.argsort()[::-1]
            print(top_k)
            for node_id in top_k:     
                #获取分类名称
                human_string = id_to_string(node_id)
                #获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()
            #显示图片
            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
```

测试结果如下：（用实验室电脑，显卡 GTX 1080ti 跑的）

``` xml
retain\images\animal.jpg
[0 1 3 4 2]
animal (score = 0.78572)
flower (score = 0.07398)
house (score = 0.06210)
plane (score = 0.05573)
guitar (score = 0.02246)
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012202426.png)

``` xml
E:\Python-projects\mnist\retrain\images\flower.jpg
[1 0 4 3 2]
flower (score = 0.93735)
animal (score = 0.02391)
plane (score = 0.01805)
house (score = 0.01259)
guitar (score = 0.00810)
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012202432.png)

``` xml
E:\Python-projects\mnist\retrain\images\guitar.jpg
[2 4 0 3 1]
guitar (score = 0.98568)
plane (score = 0.00405)
animal (score = 0.00355)
house (score = 0.00346)
flower (score = 0.00325)
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012202436.png)

``` xml
E:\Python-projects\mnist\retrain\images\house.jpg
[3 1 0 4 2]
house (score = 0.94151)
flower (score = 0.02240)
animal (score = 0.01652)
plane (score = 0.01133)
guitar (score = 0.00824)
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012202439.png)

``` xml
E:\Python-projects\mnist\retrain\images\plane.jpg
[4 0 2 1 3]
plane (score = 0.96458)
animal (score = 0.01224)
guitar (score = 0.00790)
flower (score = 0.00771)
house (score = 0.00758)
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012202441.png)

### 三、从头开始训练新的模型

上节介绍的使用 tensorflow 用已经训练好的模型进行微调，然后创建我们图片分类任务的模型。这节介绍怎么从头训练新的模型。

还是以上节图片分类任务为例，训练图片还是原来这些，简单点，这次每个分类中的图片减少到 300 张，这样总共 1500（=5*300）张图片（1000 张为训练集，500 张为测试集，下面会讲到）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-32645120.jpg)

!!!准备工作：在 https://github.com/tensorflow/models 提供了很多官方的模型，这节我们要用的 slim 模型，先下载 slim（新版中 slim 路径位置已经处在：[models](https://github.com/tensorflow/models)/[research](https://github.com/tensorflow/models/tree/master/research)/[slim](https://github.com/tensorflow/models/tree/master/research/slim)，不是视频中的 models 目录下）。slim 文件如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-42918227.jpg)

第一个步骤：首先要对图片进行预处理，在这里其实就是生成一些`.tfrecord`文件，它是 TensorFlow 官方提供的一种的文件类型，底层是 protobuf，简单讲就是一种二进制存储方式，Google 开源的。有兴趣了解下：[Protobuf简介和使用](https://www.jianshu.com/p/5ea08c6b7031)。过程大概就是：先把图片转换成`.tfrecord`格式文件，在训练模型的过程中，调用`.tfrecord`文件训练。

完整代码如下：（对应代码：`9-2生成tfrecord.py`，代码中数据集路径，标签文件名有作修改）

``` python
# coding: utf-8

import tensorflow as tf
import os
import random
import math
import sys

#验证集数量
_NUM_TEST = 500
#随机种子
_RANDOM_SEED = 0
#数据块
_NUM_SHARDS = 5
#数据集路径
DATASET_DIR = "D:/retrain/data/"
#标签文件名字
LABELS_FILENAME = "D:/retrain/output_labels.txt"

#定义tfrecord文件的路径+名字
def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'image_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)

#判断tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        for shard_id in range(_NUM_SHARDS):
            #定义tfrecord文件的路径+名字
            output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
        if not tf.gfile.Exists(output_filename):
            return False
    return True
```



``` python
#获取所有文件以及分类
def _get_filenames_and_classes(dataset_dir):
    #数据目录
    directories = []
    #分类名称
    class_names = []
    for filename in os.listdir(dataset_dir):
        #合并文件路径
        path = os.path.join(dataset_dir, filename)
        #判断该路径是否为目录
        if os.path.isdir(path):
            #加入数据目录
            directories.append(path)
            #加入类别名称
            class_names.append(filename)

    photo_filenames = []
    #循环每个分类的文件夹
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            #把图片加入图片列表
            photo_filenames.append(path)

    return photo_filenames, class_names
```

> 解释：①文件夹名称相当于分类；②得到的图片路径为绝对路径

``` python
def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, image_format, class_id):
    #Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
    }))

def write_label_file(labels_to_class_names, dataset_dir,filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))
```



``` python
#把数据转为TFRecord格式
def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    assert split_name in ['train', 'test']
    #计算每个数据块有多少数据
    num_per_shard = int(len(filenames) / _NUM_SHARDS)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                #定义tfrecord文件的路径+名字
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    #每一个数据块开始的位置
                    start_ndx = shard_id * num_per_shard
                    #每一个数据块最后的位置
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        try:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
                            sys.stdout.flush()
                            #读取图片
                            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                            #获得图片的类别名称
                            class_name = os.path.basename(os.path.dirname(filenames[i]))
                            #找到类别名称对应的id
                            class_id = class_names_to_ids[class_name]
                            #生成tfrecord文件
                            example = image_to_tfexample(image_data, b'jpg', class_id)
                            tfrecord_writer.write(example.SerializeToString())
                        except IOError as e:
                            print("Could not read:",filenames[i])
                            print("Error:",e)
                            print("Skip it\n")
                            
    sys.stdout.write('\n')
    sys.stdout.flush()
```

> 解释：`num_per_shard = int(len(filenames) / _NUM_SHARDS)`把数据块做了个切分，其实该例子来说可以不用切分的。什么时候切分和不切分呢？若数据量比较小，可以不用切分，存储在一个`.tfrecord`文件即可，若数据量比较大，如几百 G 大小数据，则选择切分为好。上面代码对数据进行了 5 个切分。

``` python
if __name__ == '__main__':
    #判断tfrecord文件是否存在
    if _dataset_exists(DATASET_DIR):
        print('tfcecord文件已存在')
    else:
        #获得所有图片以及分类
        photo_filenames, class_names = _get_filenames_and_classes(DATASET_DIR)
        #把分类转为字典格式，类似于{'house': 3, 'flower': 1, 'plane': 4, 'guitar': 2, 'animal': 0}
        class_names_to_ids = dict(zip(class_names, range(len(class_names))))

        #把数据切分为训练集和测试集
        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[_NUM_TEST:]
        testing_filenames = photo_filenames[:_NUM_TEST]

        #数据转换
        _convert_dataset('train', training_filenames, class_names_to_ids, DATASET_DIR)
        _convert_dataset('test', testing_filenames, class_names_to_ids, DATASET_DIR)

        #输出labels文件
        labels_to_class_names = dict(zip(range(len(class_names)), class_names))
        write_label_file(labels_to_class_names, DATASET_DIR)
```

> 得到 5 个文件夹下的所有图片文件，打乱，再把数据切分为训练集和测试集，再把数据转换为`.tfrecord`格式文件。

执行程序：

``` xml
>>> Converting image 1000/1000 shard 4
>>> Converting image 315/500 shard 3
```

解释：第一行表示训练集的转换，第二行表示测试集的转换。

执行完之后，D:/retrain/data/ 目录下就会多出这么些文件：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-30734712.jpg)

前面第一到第五个文件为测试集数据块，第六到第十个文件训练集数据块。打开 labels.txt，内容如下：

``` xml
0:animal
1:flower
2:guitar
3:house
4:plane
```

在下载的 slim 中找到 datasets 文件夹，新建`myimages.py`文件，如何写？——参考该文件夹其他的文件，主要修改的地方有这么几处：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-59524967.jpg)

再打开 datasets 文件夹下的 `dataset_factory.py`文件，添加：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-11-14643042.jpg)

再编写个批处理文件`train.bat`，内容如下：

``` xml
python D:/Tensorflow/slim/train_image_classifier.py ^
--train_dir=D:/Tensorflow/slim/model ^
--dataset_name=myimages ^
--dataset_split_name=train ^
--dataset_dir=D:/Tensorflow/slim/images ^
--batch_size=10 ^
--max_number_of_steps=10000 ^
--model_name=inception_v3 ^
pause
```

解释下：

- 第一行表示：使用 slim 的`train_image_classifier.py` 程序
- train_dir：训练后模型的保存位置，自己指定想要的位置就行
- dataset_name：指定在 datasets 文件夹下新建的`myimages.py`文件名称
- dataset_split_name：表示用到的是训练集
- dataset_dir：训练图片存放的位置
- batch_size：批次的大小
- max_number_of_steps：训练的步数
- model_name：使用到的模型

### 补充：tfrecord文件制作

参考：[TensorFlow 学习（二） 制作自己的TFRecord数据集，读取，显示及代码详解 - CSDN](<https://blog.csdn.net/miaomiaoyuan/article/details/56865361>)