MNIST 数据集来自美国国家标准与技术研究所，National Institute of Standards and Technology （NIST）。 训练集（training set）由来自 250 个不同人手写的数字构成，其中 50% 是高中学生，50% 来自人口普查局（the Census Bureau）的工作人员。测试集（test set）也是同样比例的手写数字数据。

很多的深度学习框架都有以 MNIST 为数据集的 Demo，MNIST 是很好的手写数字数据集。在网上很容易找到资源，但是下载下来的文件并不是普通的图片格式。不转换为图片格式也可以用。但有时，我们希望得到可视化的图片格式。（PS：MNIST 数据集里的每张图片大小为`28*28`像素，可以用`28*28`的大小的数组来表示一张图片。标签用大小为 10 的数组来表示。）

MNIST 数据集包含 4 个文件：

``` xml
train-images-idx3-ubyte: training set images	训练集-图片（9.9 MB, 解压后 47 MB, 包含 60000 个样本）
train-labels-idx1-ubyte: training set labels	训练集-标签（29 KB, 解压后 60 KB, 包含 60000 个标签） 
t10k-images-idx3-ubyte:  test set images	测试集-图片（1.6 MB, 解压后 7.8 MB, 包含 10000 个样本） 
t10k-labels-idx1-ubyte:  test set labels	测试集-标签（5KB, 解压后 10 KB, 包含 10000 个标签）
```

在[MNIST](http://yann.lecun.com/exdb/mnist/)网站对数据集的介绍中能看到如下说明：

**TRAINING SET LABEL FILE (train-labels-idx1-ubyte):**

``` xml
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label

The labels values are 0 to 9.
```

**TRAINING SET IMAGE FILE (train-images-idx3-ubyte):**

``` xml
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
```

**TEST SET LABEL FILE (t10k-labels-idx1-ubyte):**

``` xml
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  10000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label

The labels values are 0 to 9.
```

**TEST SET IMAGE FILE (t10k-images-idx3-ubyte):**

``` xml
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  10000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 
```

我们先看训练标签集`train-labels-idx1-ubyte`，官网说法，训练集是有 60000 个用例的，也就是说这个文件里面包含了 60000 个标签内容，每一个标签的值为 0 到 9 之间的一个数。

按上面数据结构来看，我们先解析每一个属性的含义，offset 代表了字节偏移量，也就是这个属性的二进制值的偏移是多少；type 代表了这个属性的值的类型；value 代表了这个属性的值是多少；description 是对这个的说明；所以呢，这里对上面的进行一下说明，它的说法是：“从第 0 个字节开始有一个 32 位的整数，它的值是  0x00000801，它是一个魔数；从第 4 个字节开始有一个 32 位的整数，它的值是 60000，它代表了数据集的数量；从第 8 个字节开始有一个 unsigned byte，它的值是 ??，是一个标签值….”。

再看训练图片集，其解说与上面的标签文件类似，但是这里还要补充说明一下，在 MNIST 图片集中，所有的图片都是`28×28`的，也就是每个图片都有`28×28`个像素；我们的`train-images-idx3-ubyte`文件中偏移量为 0 字节处有一个 4 字节的数为 0000 0803 表示魔数；接下来是 0000 ea60 值为 60000 代表容量，接下来从第 8 个字节开始有一个 4 字节数，值为 28 也就是 0000 001c，表示每个图片的行数；从第 12 个字节开始有一个 4 字节数，值也为 28，也就是 0000 001c 表示每个图片的列数；从第 16 个字节开始才是我们的像素值。每 784 个字节代表一幅图片 。

补充说明：在图示中我们可以看到有一个 MSB first，其全称是”Most Significant Bit first”，相对称的是一个 LSB first，“Least Significant Bit”；MSB first 是指最高有效位优先，也就是我们的大端存储，而 LSB 对应小端存储。关于大端小端，可参考：[详解大端模式和小端模式](https://blog.csdn.net/ce123_zhouwei/article/details/6971544)。

文件的格式可以理解为一个很长的一维数组，在`train-images-idx3-ubyte`中：

> 可以看出，第一个数为 32 位的整数（魔数，图片类型的数），第二个数为 32 位的整数（图片的个数），第三和第四个也是 32 位的整数（分别代表图片的行数和列数），接下来的都是一个字节的无符号数（即像素，值域为0~255）。

测试图像（rain-images-idx3-ubyte）与训练图像（train-images-idx3-ubyte）由 5 部分组成：

| 32bits int   | 32bits int | 32bits int | 32bits int | 像素值 |
| :----------- | ---------- | ---------- | ---------- | ------ |
| magic number | 图像个数   | 图像高度28 | 图像宽度28 | pixels |

测试标签（t10k-labels-idx1-ubyte）与训练标签（train-labels-idx1-ubyte）由 3 部分组成：

| 32bits int   | 32bits int | 标签   |
| ------------ | ---------- | ------ |
| magic number | 图像个数   | labels |

1、对于训练集数据：

``` python
import numpy as np
import struct
 
from PIL import Image
import os
 
data_file = 'somePath/train-images.idx3-ubyte' #需要修改的路径
# It's 47040016B, but we should set to 47040000B
data_file_size = 47040016
data_file_size = str(data_file_size - 16) + 'B'
 
data_buf = open(data_file, 'rb').read()
 
magic, numImages, numRows, numColumns = struct.unpack_from(
    '>IIII', data_buf, 0)
datas = struct.unpack_from(
    '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(
    numImages, 1, numRows, numColumns)
 
label_file = 'somePath/train-labels.idx1-ubyte' #需要修改的路径
 
# It's 60008B, but we should set to 60000B
label_file_size = 60008
label_file_size = str(label_file_size - 8) + 'B'
 
label_buf = open(label_file, 'rb').read()
 
magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from(
    '>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)
 
datas_root = '/somePath/mnist_train' #需要修改的路径
if not os.path.exists(datas_root):
    os.mkdir(datas_root)
 
for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)
 
for ii in range(numLabels):
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    file_name = datas_root + os.sep + str(label) + os.sep + \
        'mnist_train_' + str(ii) + '.png'
    img.save(file_name)
```

PS：代码中的`’>IIII’`表示使用大端规则，读取四个整形数（Integer），如果要读取一个字节，则可以用`’>B’`（当然，这里用没用大端规则都是一样的，因此只有两个或两个以上的字节才有用）。

最后会生成：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181101205405.png)

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181101205605.png)

2、对于测试数据集：

``` python
import numpy as np
import struct
 
from PIL import Image
import os
 
data_file = 'somePath/t10k-images.idx3-ubyte' #需要修改的路径
 
# It's 7840016B, but we should set to 7840000B
data_file_size = 7840016
data_file_size = str(data_file_size - 16) + 'B'
 
data_buf = open(data_file, 'rb').read()
 
magic, numImages, numRows, numColumns = struct.unpack_from(
    '>IIII', data_buf, 0)
datas = struct.unpack_from(
    '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(
    numImages, 1, numRows, numColumns)
 
label_file = 'somePath/t10k-labels.idx1-ubyte'#需要修改的路径
 
# It's 10008B, but we should set to 10000B
label_file_size = 10008
label_file_size = str(label_file_size - 8) + 'B'
 
label_buf = open(label_file, 'rb').read()
 
magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from(
    '>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)
 
datas_root = 'somePath/mnist_test' #需要修改的路径
 
if not os.path.exists(datas_root):
    os.mkdir(datas_root)
 
for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)
 
for ii in range(numLabels):
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    file_name = datas_root + os.sep + str(label) + os.sep + \
        'mnist_test_' + str(ii) + '.png'
    img.save(file_name)
```



参考文章：

- [使用转换mnist数据库保存为bmp图片](http://www.itboth.com/d/YVB7Fb/bmp-python-mnist)
- [MNIST数据集转换为图像](https://blog.csdn.net/u012507022/article/details/51376626)
- [如何用python解析mnist图片](https://blog.csdn.net/u014046170/article/details/47445919)
- [（超详细）读取mnist数据集并保存成图片](https://blog.csdn.net/YF_Li123/article/details/76710028)
- [使用Python将MNIST数据集转化为图片](https://blog.csdn.net/qq_32166627/article/details/52640730)

延伸：

- [如何将图片转换为mnist格式的数据？](https://www.zhihu.com/question/55963897)
- 通过 TensorFlow API 接口导出 MNIST 图片的 Python 代码：[导出MNIST的数据集](https://www.cnblogs.com/jyxbk/p/7773295.html)