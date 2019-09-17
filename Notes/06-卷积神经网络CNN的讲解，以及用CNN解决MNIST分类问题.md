## 卷积神经网络CNN，用CNN解决MNIST分类问题

### 一、卷积神经网络

#### 认识卷积神经网络

相关资料：

- [李理：详解卷积神经网络](https://blog.csdn.net/qunnie_yi/article/details/80127218)
- 机器之心：[从入门到精通：卷积神经网络初学者指南](https://www.jiqizhixin.com/articles/2016-08-01-3)
- [图文并茂地讲解卷积神经网络](https://mp.weixin.qq.com/s/ixwEVn_WMkH28w5aYITnBw)
- charlotte77博客：[【深度学习系列】卷积神经网络CNN原理详解(一)——基本原理](https://www.cnblogs.com/charlotte77/p/7759802.html)
- 知乎：[能否对卷积神经网络工作原理做一个直观的解释？](https://www.zhihu.com/question/39022858/answer/194996805)
- ......

上面一些文章讲解的很清楚。借此，顺带多絮叨几句。

到底深度学习是什么？有什么特点？下面举例来理解下这玩意：

> 假设有一张图，要做分类，传统方法需要手动提取一些特征，比如纹理啊，颜色啊，或者一些更高级的特征。然后再把这些特征放到像随机森林等分类器，给到一个输出标签，告诉它是哪个类别。而深度学习是输入一张图，经过神经网络，直接输出一个标签。特征提取和分类一步到位，避免了手工提取特征或者人工规则，从原始数据中自动化地去提取特征，是一种端到端（end-to-end）的学习。相较于传统的方法，深度学习能够学习到更高效的特征与模式。
>
> ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-28008326.jpg)

应用到计算机视觉方向来说，简单来说就是深度学习可以自己学习到图像特征（其背后数学层面来看，也就是学到一个含非常多参数的函数），而不要我们自己去提取特征，即，不要我们去定义具有怎样特征才是猫，比如是否头部近圆形，颜面部短，耳呈三角形这样的特征才是猫，我们不用关心，深度学习能自动学习到特征（当然其实我们也不知道它到底学到了什么特征，所以被很多人称为「黑匣子」，可以看这篇文章 [1.1.1 什么是神经网络](https://blog.csdn.net/jiangjunshow/article/details/77368314) 体会下为什么这么说）。

传统经典网络存在的问题：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-94425588.jpg)

- 权值太多，计算量太大
- 权值太多，需要大量样本进行训练

经验之谈：样本数量最好是参数数量的 5—30 倍。数据量小而模型参数过的多容易出现过拟合现象。

#### 局部感受野

> 1962 年哈佛医学院神经生理学家 Hubel 和 Wiesel 通过对猫视觉皮层细胞的研究，提出了感受野（receptive field）的概念，1984 年日本学者 Fukushima 基于感受野概念提出的神经认知机（neocognitron）可以看作是卷积神经网络的第一个实现网络，也是感受野概念在人工神经网络领域的首次应用。

怎么理解局部感受野？举例来说。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-54428842.jpg)

如上是一个全连接神经网络，全连接指的是：对 n-1 层和 n 层而言，n-1 层的任意一个节点，都和第 n 层所有节点有连接。明显地，网络很大的时候，参数很多，训练速度会很慢。

但在卷积网络里，我们把输入看成二维神经元，它的每一个神经元对应于图片在这个像素点的强度（灰度值），如下图所示：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-82756302.jpg)

把输入像素连接到隐藏层的神经元（怎么做的呢？——先把“图像所有像素值拉直”，再连接到隐藏层的神经元，见下图体会）。但是我们这里**不再把输入的每一个像素都连接到隐藏层的每一个神经元**，与之不同，我们把很小的相临近的区域内的输入连接在一起。具体的来讲，隐藏层的每一个神经元都会与输入层一个很小的区域（比如一个 3×3 的区域，也就是 9 个像素点）相连接。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-45145457.jpg)

​					（*上图来源台湾大学李宏毅老师《深度学习》PPT内容*）

输入图像的这个区域叫做那个隐藏层神经元的局部感知域。这是输入像素的一个小窗口。每个连接都有一个可以学习的权重，此外还有一个 bias（偏置）。对于最右上的那个神经元（即，Filter——称过滤器、或滤波器、或卷积核）你可以想象成用来分析这个局部感知域的。

然后在整个输入图像上滑动这个局部感知域，这里就会涉及到步伐的问题了。我们可以一次移动一个像素（这个移动的值叫 strides），也可以一次移动不止一个像素。

说明：如果需要让图像在经过这样一次卷积处理后尺寸可以不变小，可以使用 padding，简单讲，就是把图片像素的边边角角拼一段像素上去，有两种方式，一种是填 0，另一种是将边边角角的像素直接复制一个填进去。那 padding 要拼多少像素可以根据 filter 大小来定，filter 越大，需要拼的就越多。padding 是不是一定比不做效果好，这个视情况而定，多炼丹才知道。

**卷积过程的 padding：**

另外关于 padding 有两种类型：

- SAME PADDING
- VALID PADDING

关于两者区别，下面摘录知乎一个回答：

> 唐突做一下解释：在卷积核移动逐渐扫描整体图时候，因为步长的设置问题，可能导致剩下未扫描的空间不足以提供给卷积核的，大小扫描 比如有图大小为`5x5`，卷积核为`2x2`，步长为 2，卷积核扫描了两次后，剩下一个元素，不够卷积核扫描了，这个时候就在后面补零，补完后满足卷积核的扫描，这种方式就是 same。如果说把刚才不足以扫描的元素位置抛弃掉，就是 valid 方式。
>
> 知乎：[TensorFlow中padding的SAME和VALID两种方式有何异同？](https://www.zhihu.com/question/60285234)

SAME PADDING：可能会给平面外部补 0，卷积窗口采样后得到一个跟原来平面大小相同的平面。

VALID PADDING：不会超出平面外部，卷积窗口采样后得到一个比原来平面小的平面。

**池化过程的 padding：**

1）假如有一个`28x28`的平面，用`2x2`并且步长为 2 的窗口对其进行 pooling 操作：

- 使用 SAME PADDING 的方式，得到`14x14`的平面
- 使用 VALID PADDING 的方式，得到`14x14`的平面

2）假如有一个`2x3`的平面，用`2x2`并且步长为 2 的窗口对其进行 pooling 操作

- 使用 SAME PADDING 的方式，得到`1x2`的平面
- 使用 VALID PADDING 的方式，得到`1x1`的平面

①补充1：以上关于 padding 的解释看看就好，不要深究！还是看看在 TensorFlow 中 padding 的实现。——**关于卷积核池化及 padding 在 TensorFlow 中的操作是怎样的**，请阅读本文 【[补充内容：关于TensorFlow中的CNN卷积和池化的操作](#补充内容关于tensorflow中的cnn卷积和池化的api详解)】小节以及 [TensorFlow的API详解和记录.md](../other/[整理]TensorFlow的API详解和记录.md) 中的内容。

②补充2： 关于 CNN 中的 padding，表示有在网上找了些博客看看，现摘入如下，以便随时查阅。

**Convolution Arithmetic**（卷积运算）

输入的尺寸为 i，卷积核大小为 k，strides 的大小为 s，padding 的大小为 p，输出的尺寸为 o，只考虑卷积核和输入的 x 和 y 相等的情况。

**No zero padding，unit strides：** 没有 0 填充，步伐为 1

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105165632.png)

> 输出的尺寸大小：o = (i - k) + 1

**Zero padding，unit strides：** 有 0 填充，步伐为 1

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105165750.png)

> 输出的尺寸大小：o = (i - k) + 2p + 1

**Half (same) padding：** 在这里输入与输出的大小一样，这是一个期望的特性

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105165835.png)

> 这种方式的卷积要进行 padding，并且目的是保证输出和输入具有相同的尺寸。由于卷积过程中使用的卷积核一半大小为奇数，所以为了保证：(i-k)+2p+1 = i，则 p=(k-1)/2=(k/2) 的向下取整。

**Full padding：** 当需要输出比输入更大时

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105165907.png)

> padding 的大小为 k-1

**No zero padding，non-unit strides：** 没有 0 填充，步伐不为 1

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105165949.png)

> 输出的尺寸大小：o = ((i-k)/s)向下取整 + 1

**Zero padding，non-unit strides：** 有 0 填充，步伐不为 1

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105170017.png)

> 输出的尺寸大小：o = ((i-k+2*p)/s)向下取整 + 1

参考资料：

- *[A guide to convolution arithmetic for deep learning学习笔记](https://blog.csdn.net/cdknight_happy/article/details/78898791)*
- *[padding](https://blog.csdn.net/jyli2_11/article/details/72784573)*
- *Vincent Dumoulin, Francesco Visin——[A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)， 2016-3-24* 		    arxiv：1603.07285
- *GitHub 地址：[conv_arithmetic](https://github.com/ysglh/conv_arithmetic)*

  > *A technical report on convolution arithmetic in the context of deep learning.*
  >
  > PS1：该项目下有卷积 Convolution、转置卷积 Transposed convolution、空洞卷积 Dilated convolution 以及不同 padding、strides 情况下的动画。

更多关于卷积和转置卷积的理解来看看这篇文章[CNN中卷积层与转置卷积层的关系（转置卷积又称反卷积、分数步长卷积）](https://blog.csdn.net/dugudaibo/article/details/83109814),其中，对于转置卷积中的输入中间有插入 0 的解释，可以看下文章 2.5 节内容，摘入部分：

由于转置卷积的步长是直接卷积的倒数，因此当直接卷积的步长 s>1 的时候，那么转置卷积的步长就会是分数，这也是转置卷积又称为分数步长卷积的原因。在前面例子中，我们所处理的都是直接卷积步长为 1 的例子，所以可以认为直接卷积与转置卷积的步长相等。当转置卷积的步长小于 1 的时候，我们可以通过下面的例子有一个直接的了解：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181216142438.png)

如上图是一个输入 feature map 为 5×5 ，卷积核大小为 3×3，步长 s=2 的直接卷积的转置卷积，此时的转置卷积的输入是在 2×2 的矩阵间进行插孔得到的。首先计算此时转置卷积输出的大小，我们发现与之前的计算方法是一样的：W1=S(W2−1)−2P+F=2×(2−1)−2×0+3=5，果然通过之前推导出的公式计算出了与上图相同的结果，这时我们计算下转置卷积中 padding 的大小：P^T=F−P−1=3−0−1=2。

很明显 padding 的计算结果也是符合上面的公式要求的。之后就是最关键的部分了，如何体现出步长是分数步长。在原始的卷积中插入数字 0，这使得内核以比单位步幅的速度移动慢，具体的在输入的每两个元素之间插入 s−1 个 0。所以此时转置卷积的输入尺寸大小由原来的 W2 变为 W2+(W2−1)(s−1)。

#### 权值共享

权值共享这个词最开始其实是由 LeNet5 模型提出来，在 1998 年，LeCun 发布了 LeNet 网络架构，就是下面这个： 

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-9-1224071.jpg)

虽然现在大多数的说法是 2012 年的 AlexNet 网络模型是深度学习的开端，但是 CNN 的开端最早其实可以追溯到 LeNet5 模型，它的几个特性在 2010 年初的卷积神经网络研究中被广泛的使用——其中一个就是**权值共享**。

到底怎么理解权值共享呢？——举例来说，所谓的权值共享就是说，给一张输入图片，用一个 filter 去扫这张图，filter 里面的数就叫权重，这张图每个位置是被同样的 filter 扫的，所以权重是一样的，也就是共享，说白了，就是整张图片在使用同一个 filter 的参数。

比如一个`3x3x1`的 filter（卷积核，另说明下：这里的 `x1` 表示为单通道图像），这个 filter 内 9 个的参数被整张图共享，而不会因为 filter 在图像上滑动后位置的不同而改变 filter 内的权系数，说的再直白一些，就是用一个 filter 不改变其内权系数的情况下卷积处理整张图片（当然 CNN 中每一层不会只有一个 filter 的，这样说只是为了方便解释而已，实际中，每层会有多个不同 filter，为了提取图像不同的特征）。下图为台大李宏毅老师《深度学习》PPT 某页内容，可以对照着理解下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-9-37228310.jpg)

​				（*上图来源台湾大学李宏毅老师《深度学习》PPT内容*）

参考：

- 知乎：[如何理解卷积神经网络中的权值共享？](https://www.zhihu.com/question/47158818)
- [如何理解卷积神经网络中的权值共享](https://blog.csdn.net/chaipp0607/article/details/73650759)

推荐 B 站视频：[李宏毅-Convolutional Neural Network（CNN）-卷积神经网络](https://www.bilibili.com/video/av23593949/)

#### 卷积

单通道图像卷积过程（如下使用了一个卷积核卷积）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-9-30544234.jpg)

动态图过程：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-9-3764121.jpg)

三通道（R、G、B ，可以理解为深度为 3）图像卷积过程（如下使用了两个卷积核卷积）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-9-43496624.jpg)

多个卷积核卷积用来提取不同特征：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-9-87362924.jpg)

#### 池化(Pooling)

pooling 层可以非常有效地缩小图片的尺寸，显著减少参数数量，但 pooling 的目的并不仅在于此。pooling 目的是为了保持某种不变性（旋转、平移、伸缩等），常用的有 mean-pooling，max-pooling 和 Stochastic-pooling 三种。

1）mean-pooling（平均池化）：即对邻域内特征点只求平均，对背景保留更好

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-9-73494598.jpg)

2）max-pooling（最大池化）：对邻域内特征点取最大，对纹理提取更好

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-9-20901289.jpg)

3）Stochastic-pooling：介于两者之间，通过对像素点按照数值大小赋予概率，再按照概率进行亚采样，在平均意义上，与 mean-pooling 近似，在局部意义上，则服从 max-pooling 的准则

#### 注：多通道图像池化

在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按通道相加。这意味着池化层的输出通道数与输入通道数相等。

> ——from：[5.4. 池化层 — 《动手学深度学习》 文档](<https://zh.d2l.ai/chapter_convolutional-neural-networks/pooling.html>)

#### 补充内容：参数数量和连接数的计算

先看下掘金上这篇文章：【[全连接网络到卷积神经网](https://juejin.im/post/5ae01a6e6fb9a07a9a108441)】，关于全连接网络、卷积神经网络讲解的挺清楚的，看完相信会加深对相关内容理解。

但关于文章中对参数个数的计算，我觉得是有问题的，并没有把偏置 bias 参数算进去。比如，对于全连接网络，第 i 层的每个神经元与第 i-1 层中的所有神经元相连，假设第 i-1 层有 9 个神经元，第 i 层有 16 个神经元：

![](https://user-gold-cdn.xitu.io/2018/4/25/162fb678cc8b81c9?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

可以看到，第 i-1 层每个神经元都与第 i 层相连接。那么该两层网络总共的参数数量应该是（加上 bias）：9x16+16=160 个参数（权重）。用文字表达就是：`输入长度`x`该层神经元个数`+`偏置（该层神经元个数）`=`该层参数数量`。

那么对于卷积神经网络的参数数量的计算呢？我们假设输入层的维度为 32x32x3，第一层卷积层使用尺寸为 5x5，深度为 16 的过滤器，那么这个卷积层的参数个数为：5x5x3x16=1216 个。可以回头看下三通道（R、G、B）图像卷积过程的 Gif 动图。用文字表达一下就是：`卷积核的宽`x`卷积核的长`x`输入的通道数`x`卷积核的个数`+`卷积核个数`=`该层的总参数`。

**我们以最经典的 LeNet-5 例子来逐层分析各层的参数及连接个数。**

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-9-1224071.jpg)

C1 层是一个卷积层，由 6 个特征图 Feature Map 构成。特征图中每个神经元与输入为 5x5 的邻域相连。特征图的大小为 28x28，这样能防止输入的连接掉到边界之外（32-5+1=28）。C1 有 156 个可训练参数（每个滤波器5x5=25个 unit 参数和一个 bias 参数，一共 6 个滤波器，共(5x5+1)x6=156个参数），共 156x(28x28)=122304 个连接。

> 或这么理解：经过 5x5 的 filter 后，下一层的节点矩阵有 28x28x6=4704 个节点，每个节点和 5x5=25 个当前节点连接，所以本层卷积层总共有 4704x(25+1)=122304 个连接。*——From：《TensorFlow实战Google深度学习框架》*

S2 层是一个下采样层，有 6 个14x14 的特征图。特征图中的每个单元与 C1 中相对应特征图的 2x2 邻域相连接。S2 层每个单元的 4 个输入相加，乘以一个可训练参数，再加上一个可训练偏置。每个单元的 2x2 感受野并不重叠，因此 S2 中每个特征图的大小是 C1 中特征图大小的 1/4（行和列各1/2）。S2 层有12（6x（1+1）=12）个可训练参数和 5880（14x14x（2x2+1）x6=5880）个连接。

C3 层也是一个卷积层，它同样通过 5x5 的卷积核去卷积层 S2，然后得到的特征 map 就只有 10x10 个神经元，但是它有 16 种不同的卷积核，所以就存在 16 个特征 map 了。 C3 中每个特征图由 S2 中所有 6 个或者几个特征 map 组合而成。为什么不把 S2 中的每个特征图连接到每个 C3 的特征图呢？原因有 2 点。第一，不完全的连接机制将连接的数量保持在合理的范围内。第二，也是最重要的，其破坏了网络的对称性。由于不同的特征图有不同的输入，所以迫使他们抽取不同的特征（希望是互补的）。

例如，存在的一个方式是：C3 的前 6 个特征图以 S2 中 3 个相邻的特征图子集为输入。接下来 6 个特征图以 S2 中 4 个相邻特征图子集为输入。然后的 3 个以不相邻的 4 个特征图子集为输入。最后一个将 S2 中所有特征图为输入。这样 C3 层有1516（6x（3x25+1）+6x（4x25+1）+3x（4x25+1）+（25x6+1）=1516）个可训练参数和151600（10x10x1516=151600）个连接。

 S4 层是一个下采样层，由 16 个 5x5 大小的特征图构成。特征图中的每个单元与 C3 中相应特征图的 2x2 邻域相连接，跟 C1 和 S2 之间的连接一样。S4 层有 32 个可训练参数（每个特征图 1 个因子和一个偏置 16x（1+1）=32）和 2000（16x（2x2+1）x5x5=2000）个连接。

C5 层是一个卷积层，有 120 个特征图。每个单元与 S4 层的全部 16 个单元的 5x5 邻域相连。由于 S4 层特征图的大小也为 5x5（同滤波器一样），故 C5 特征图的大小为 1x1（5-5+1=1）：这构成了 S4 和 C5 之间的全连接。之所以仍将 C5 标示为卷积层而非全相联层，是因为如果 LeNet-5 的输入变大，而其他的保持不变，那么此时特征图的维数就会比 1x1 大。C5 层有 48120（120x（16x5x5+1）=48120 由于与全部 16 个单元相连，故只加一个偏置）个可训练连接。

F6 层有 84 个单元（之所以选这个数字的原因来自于输出层的设计），与 C5 层全相连。有10164（84x(120x(1x1)+1)=10164）个可训练参数。如同经典神经网络，F6 层计算输入向量和权重向量之间的点积，再加上一个偏置。然后将其传递给sigmoid函数产生单元i的一个状态。

最后，输出层由欧式径向基函数（Euclidean Radial Basis Function）单元组成，每类一个单元，每个有 84 个输入。

### 二、编码实现

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-9-8454243.jpg)

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-9-24953964.jpg)

定义 weight、bias；

卷积、激活、池化、下一层；

然后接 2 个全连接层，softmax，交叉熵、loss

（代码对应：`6-1卷积神经网络应用于MNIST数据集分类.py`，有修改——增加很多命名空间 scope）

``` python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布
    return tf.Variable(initial)

# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(x, W):
    # x input tensor of shape '[batch,in_height,in_width,in_channles]'
    # W filter / kernel tensor of shape [filter_height,filter_width,in_channels,out_channels]
    # `strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    # padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 2d的意思是二维的卷积操作

# 池化层
def max_pool_2x2(x):
    # ksize [1,x,y,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])  # 28*28
y = tf.placeholder(tf.float32, [None, 10])

# 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷积层的权值和偏置
W_conv1 = weight_variable([5, 5, 1, 32])  # 5*5的采样窗口，32个卷积核从1个平面抽取特征
b_conv1 = bias_variable([32])  # 每一个卷积核一个偏置值

# 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 进行max-pooling

# 初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable([5, 5, 32, 64])  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
b_conv2 = bias_variable([64])  # 每一个卷积核一个偏置值

# 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # 进行max-pooling

# 28*28的图片第一次卷积后还是28*28（数组变小了，但是图像大小不变），第一次池化后变为14*14
# 第二次卷积后为14*14（卷积不会改变平面的大小），第二次池化后变为了7*7
# 进过上面操作后得到64张7*7的平面

# 初始化第一个全连接层的权值
W_fc1 = weight_variable([7 * 7 * 64, 1024])  # 上一层有7*7*64个神经元，全连接层有1024个神经元
b_fc1 = bias_variable([1024])  # 1024个节点

# 把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 初始化第二个全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))
```

PS：我的笔记本跑不动啊o(╥﹏╥)o  显卡不支持深度学习框架。

显卡是否支持深度学习得看是否支持 CUDA（Compute Unified Device Architecture），如何查看显卡型号是否支持 CUDA：[TensorFlow-GPU：查看电脑显卡型号是否支持CUDN,以及相关软件下载与介绍](https://www.cnblogs.com/chamie/p/8707420.html)

遂还是拿实验室电脑，显卡 1080ti GPU 上跑吧，训练和测试过程如下：

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0, Testing Accuracy= 0.8637
Iter 1, Testing Accuracy= 0.9654
Iter 2, Testing Accuracy= 0.9733
Iter 3, Testing Accuracy= 0.9783
Iter 4, Testing Accuracy= 0.9829
Iter 5, Testing Accuracy= 0.9832
Iter 6, Testing Accuracy= 0.9847
Iter 7, Testing Accuracy= 0.9873
Iter 8, Testing Accuracy= 0.9867
Iter 9, Testing Accuracy= 0.988
Iter 10, Testing Accuracy= 0.9901
Iter 11, Testing Accuracy= 0.9908
Iter 12, Testing Accuracy= 0.989
Iter 13, Testing Accuracy= 0.991
Iter 14, Testing Accuracy= 0.9903
Iter 15, Testing Accuracy= 0.9911
Iter 16, Testing Accuracy= 0.9909
Iter 17, Testing Accuracy= 0.9916
Iter 18, Testing Accuracy= 0.9913
Iter 19, Testing Accuracy= 0.9901
Iter 20, Testing Accuracy= 0.991
```

使用传统的神经网络我们可能只能达到 98% 点多的准确率，可以看到，使用卷积神经网络之后，我们可以达到 99% 的准确率，虽说差了百分之一，但是接近 100%，应该说算是比较大的提升。

#### 补充内容：关于TensorFlow中的CNN卷积和池化的API详解

**（1）卷积**

TensorFlow 中的卷积一般是通过`tf.nn.conv2d()`函数实现的具体可以查看官网：https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

如：`tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')`

```xml
# tf.nn.conv2d非常方便实现卷积层前向传播算法
# 第一个输入为当前层的节点矩阵，这个矩阵为四维矩阵，第一维对应一个输入batch，后三维为节点矩阵
# 例如，input[0,:,:,:]表示第一张图片，input[1,:,:,:]为第二张图片
# 第二个输入为卷积层的权重，第三个输入为不同维度上的步长
# 第三个输入提供的是一个长度为4的数组，但是数组第一位和第四位一定要是1，因为卷积层的步长只对矩阵的长和宽有效
# 第四个输入时填充的方法，TensorFlow只提供两种选择，SAME为全0填充，VALID为不添加
```

定义如下：

``` python
def conv2d(input, 
           filter, 
           strides,
           padding, 
           use_cudnn_on_gpu=None,
           data_format=None, 
           name=None)
```

其中参数分别为： 

- 第一个参数为当前层的矩阵，在卷积神经网络中它是一个四维的矩阵，即 `[batch, image.size.height, image.size.width, depth]`； 
- 第二个参数为卷积核（滤波器），由 tf.get_variable 函数创建得到； 
- 第三个参数为不同维度上的步长，其实对应的是第一个参数，虽然它也是四维的，但是第一维和第四维的数一定为 1，因为我们不能间隔的选择 batch 和 depth； 
- 第四个参数为边界填充方法。



---

补充，strides：第 1，第 4 参数都为 1，中间两个参数为卷积步幅，如：`[1, 1, 1, 1]`、`[1, 2, 2, 1]`

1. 使用 VALID 方式，feature map 的尺寸为       (3,3,1,32) 卷积权重

   ```xml
   out_height=ceil(float(in_height-filter_height+1)/float(strides[1])) (28-3+1)/1= 26，(28-3+1)/2=13
   
   out_width=ceil(float(in_width-filter_width+1)/float(strides[2])) (28-3+1)/1 = 26，(28-3+1)/2=13
   
   ```

2. 使用 SAME 方式，feature map 的尺寸为     (3,3,1,32)卷积权重

   ```xml
   out_height= ceil(float(in_height)/float(strides[1]))  28/1=28，28/2=14
   
   out_width = ceil(float(in_width)/float(strides[2]))   28/1=28，28/2=14
   ```

   其中：ceil 表示为向上取整。

**（2）池化**

TensorFlow 中的池化有几种方式，举个例子，通过 tf.nn.max_pool 函数实现的具体可以查看官网：https://www.tensorflow.org/api_docs/python/tf/nn/max_pool

如：`tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')`）

``` xml
# tf.nn.max_pool函数实现了平均池化层，用法与avg_pool相似
# tf.nn.max_pool实现了最大池化层的前向传播过程，参数和conv2d类似
# ksize提供了过滤器的尺寸，数组第一位和第四位一定要是1，比较常用的是[1,2,2,1]和[1,3,3,1]
# strides提供了步长，数组第一位和第四位一定要是1
# padding提供了是否全0填充
```

定义如下：

``` python
def max_pool(value, 
             ksize, 
             strides, 
             padding, 
             data_format="NHWC", 
             name=None)
```

其中的参数：

- 第一个参数 value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是 feature map，依然是[batch, height, width, channels] 这样的 shape
- 第二个参数 ksize：池化窗口的大小，取一个四维向量，一般是 [1, height, width, 1]，因为我们不想在 batch 和 channels 上做池化，所以这两个维度设为了 1
- 第三个参数 strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是 [1, stride,stride, 1]
- 第四个参数 padding：和卷积类似，可以取 'VALID' 或者 'SAME'

补充， 

- ksize：第 1，第 4 参数都为 1，中间两个参数为池化窗口的大小，如：`[1,1,1,1]`、`[1,2,2,1]`

  实验证明：对于实际的池化后的数据尺寸，ksize没有影响，只是计算的范围不同。

- strides：第 1，第 4 参数都为 1，中间两个参数为池化窗口的步幅，如：`[1,1,1,1]`、`[1,2,2,1]`

  实验证明：对于实际的池化后的数据尺寸，strides 产生影响，具体的计算方式和卷积中的 strides 相同。

#### padding 总结

**关于 TensorFlow 中两种 padding 方式“SAME” 和 “VALID” 的到底怎么理解，**先阅读下这两篇文章：

- \[1] [TensorFlow中CNN的两种padding方式“SAME”和“VALID”](https://blog.csdn.net/wuzqChom/article/details/74785643)
- \[2] [Tensorflow中padding的两种类型SAME和VALID](https://blog.csdn.net/jiandanjinxin/article/details/77519629)

个人总结：

- 在 TensorFlow 中，对于 “VALID” 方式的  padding，按照指定步伐滑动的过程中，一个完整的卷积核覆盖不到余下窗口覆盖，则丢弃。计算方式，官方定义：

  ``` xml
  The TensorFlow Convolution example gives an overview about the difference between SAME and VALID :
  
      For the SAME padding, the output height and width are computed as:
  
      out_height = ceil(float(in_height) / float(strides[1]))
  
      out_width = ceil(float(in_width) / float(strides[2]))
  
  And
  
      For the VALID padding, the output height and width are computed as:
  
      out_height = ceil(float(in_height - filter_height + 1) / float(strides1))
  
      out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
  ```

  根据文章[1]中讲到的，对于`VALID`，**输出的形状可以这样计算：**（「 为向上取整。）![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181229200804.png)

- 在 TensorFlow 中，对于 “SAME” 方式的 padding，按照指定步伐滑动的过程中，一个完整的卷积核覆盖不到余下窗口覆盖，则补充 0 使得能覆盖到。计算方式，官方定义在上面。根据文章[1]中讲到的，对于`SAME`，**输出的形状可以这样计算：**![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181229201052.png)

**!!!注：本人在使用代码亲自实践得出的结果，和使用上面公式计算结果是一致的。** 所以计算卷积后得到的尺寸按照公式计算即可。

``` python
#keras 代码
from keras.layers import Input, Conv2D
def Test():
    input = Input([513, 513, 3])
    con1 = Conv2D(32, (5, 5), strides=(2, 2), activation="relu", padding="same")(input)
    con2 = Conv2D()(con1)
    
if __name__ == '__main__':
    Test()
    
# debug 可以看到，conv1 大小为 257x257，即 513/2 向上取整。
```

对于 “SAME” 方式的 padding，我要补充说明的是，也是我的理解：在 TensorFlow 的实现中，比如左右 padding 多少圈 0  不一定是对称的，可能是只有右边 padding 了 0，可能左右都 padding  了 0，但数量不对称。

可以看下文章[2]中例子：Input width=13，Filter width=6，Stride=5，不同的 padding 方式如下图：

  ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181229202012.png)

  其中可以看到 “SAME” 方式，在左侧 padding 了一列 0，在右侧 padding 了两列 0。

**个人理解：就是说采用 ”SAME“ 方式，在滑动过程中余下窗口元素不够的情况下，一定会 padding 一定数量的 0 以至能覆盖到余下窗口元素。**

[TensorFlow中CNN的两种padding方式“SAME”和“VALID”](<https://blog.csdn.net/wuzqChom/article/details/74785643>)：

> 让我们来看看变量 x 是一个 2×3 的矩阵，max pooling 窗口为2×2，两个维度的步长 strides=2。
>
> 第一次由于窗口可以覆盖，橙色区域做 max pooling，没什么问题，如下：
>
> ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190528103657.png)
>
> 接下来就是SAME和VALID的区别所在：由于步长为 2，当向右滑动两步之后，VALID方式发现余下的窗口不到 2×2 所以直接将第三列舍弃，而 SAME 方式并不会把多出的一列丢弃，但是只有一列了不够 2×2 怎么办？填充！
>
> ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190528103622.png)
>
> 如上图所示，SAME 会增加第四列以保证可以达到 2×2，但为了不影响原始信息，一般以 0 来填充。这就不难理解不同的 padding 方式输出的形状会有所不同了。
>
> > 当 CNN 用于文本中时，一般卷积层设置卷积核的大小为 n×k，其中k为输入向量的维度（即[n,k,input_channel_num,output_channel_num]），这时候我们就需要选择“VALID”填充方式，这时候窗口仅仅是沿着一个维度扫描而不是两个维度。可以理解为统计语言模型当中的 N-gram。



----



完成卷积神经网络，记录下准确率和 loss 率的变化，完整代码如下：（代码对应：`7-1第六周作业.py`）

``` python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

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

#### 补充内容：TensorFlow中Summary的用法

关于这里的 Summary 用法在此顺带补充些内容，方便查阅。参考【[Tensorflow学习笔记——Summary用法](https://www.cnblogs.com/lyc-seu/p/8647792.html)】

`tf.summary()` 的各类方法，能够保存训练过程以及参数分布图并在tensorboard显示。tf.summary 有诸多函数：

1. tf.summary.scalar

   用来显示标量信息，其格式为：`tf.summary.scalar(tags, values, collections=None, name=None)`

   例如：`tf.summary.scalar('mean', mean)`，一般在画 loss，accuary 时会用到这个函数。

2. tf.summary.histogram

   用来显示直方图信息，其格式为：`tf.summary.histogram(tags, values, collections=None, name=None) `

   例如：` tf.summary.histogram('histogram', var)`，一般用来显示训练过程中变量的分布情况

3. tf.summary.distribution

4. tf.summary.text

5. tf.summary.image

6. tf.summary.audio

7. tf.summary.merge_all

   merge_all 可以将所有 summary 全部保存到磁盘，以便 tensorboard 显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。格式：`tf.summaries.merge_all(key='summaries')`

8. tf.summary.FileWriter

   指定一个文件用来保存图。格式：`tf.summary.FileWritter(path,sess.graph)`，可以调用其`add_summary()`方法将训练过程数据保存在 filewriter 指定的文件中

9. tf.summary.merge

   格式：`tf.summary.merge(inputs, collections=None, name=None)`，一般选择要保存的信息还需要用到`tf.get_collection()`函数


``` python
# 初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布
    return tf.Variable(initial, name=name)
```



``` python
# 初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)
```



``` python
# 卷积层
def conv2d(x, W):
    # x input tensor of shape `[batch, in_height, in_width, in_channels]`
    # W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    # `strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    # padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
```



``` python
# 池化层
def max_pool_2x2(x):
    # ksize [1,x,y,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```



``` python
# 命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        # 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

with tf.name_scope('Conv1'):
    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')  # 5*5的采样窗口，32个卷积核从1个平面抽取特征
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')  # 每一个卷积核一个偏置值

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)  # 进行max-pooling

with tf.name_scope('Conv2'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')  # 每一个卷积核一个偏置值

    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)  # 进行max-pooling

# 28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14
# 第二次卷积后为14*14，第二次池化后变为了7*7
# 进过上面操作后得到64张7*7的平面

with tf.name_scope('fc1'):
    # 初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')  # 上一场有7*7*64个神经元，全连接层有1024个神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1')  # 1024个节点

    # 把池化层2的输出扁平化为1维
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_flat')
    # 求第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    # keep_prob用来表示神经元的输出概率
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

with tf.name_scope('fc2'):
    # 初始化第二个全连接层
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10], name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        # 计算输出
        prediction = tf.nn.softmax(wx_plus_b2)

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

# 使用AdamOptimizer进行优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    for i in range(1001):	
        # 训练模型
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        # 记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        # 记录测试集计算的参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_prob: 1.0})
            print("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) + ", Training Accuracy= " + str(train_acc))
```

**!!!注：** 先来看看上面的这部分代码，我觉得有问题！

``` python
with tf.name_scope('softmax'):
        # 计算输出
        prediction = tf.nn.softmax(wx_plus_b2)

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
```

阅读 [【TensorFlow】tf.nn.softmax_cross_entropy_with_logits的用法](https://blog.csdn.net/zj360202/article/details/78582895) 该文可以了解到 tf.nn.softmax_cross_entropy_with_logits 函数的 logits 参数传入的是未经过 softmax 的 label 值。

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

但是视频里该例子的程序，prediction 已经经历了一次 softmax 呢！

``` python
prediction = tf.nn.softmax(wx_plus_b2)
```

然后又经过了 tf.nn.softmax_cross_entropy_with_logits 函数，**这相当于经过两个 softmax 了**。（我觉得可能是视频里老师没注意到这点问题，虽然大的值的概率值还是越大，这点上倒是没影响。）

不管那么多，运行程序，结果如下：（用的实验室电脑，显卡 GTX 1080ti 跑的）

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0, Testing Accuracy= 0.1051, Training Accuracy= 0.1119
Iter 100, Testing Accuracy= 0.595, Training Accuracy= 0.5961
Iter 200, Testing Accuracy= 0.7324, Training Accuracy= 0.7365
Iter 300, Testing Accuracy= 0.7594, Training Accuracy= 0.7579
Iter 400, Testing Accuracy= 0.8423, Training Accuracy= 0.8376
Iter 500, Testing Accuracy= 0.9393, Training Accuracy= 0.9327
Iter 600, Testing Accuracy= 0.9509, Training Accuracy= 0.9468
Iter 700, Testing Accuracy= 0.9562, Training Accuracy= 0.953
Iter 800, Testing Accuracy= 0.9589, Training Accuracy= 0.9582
Iter 900, Testing Accuracy= 0.9624, Training Accuracy= 0.9584
Iter 1000, Testing Accuracy= 0.9633, Training Accuracy= 0.9617
```

程序运行完成之后会在当前程序路径下生成 logs 文件夹，logs 文件夹下会有：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012204117.png)

可视化网络训练过程：`tensorboard --logdir=logs目录的路径`

准确率：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012204758.png)

在 logs 文件夹下有两个子文件夹，对应着图中两条线，橙色对应测试集测出来的数据，蓝色对应训练集训练出来的数据，可以看到，两条线非常接近，代表模型没有欠拟合和过拟合现象。如果是过拟合情况，那么蓝色的线就会比较高，橙色的线就会比较低。

交叉熵：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012205455.png)

网络结构：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012205527.png)

fc2 内部：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181012210012.png)

