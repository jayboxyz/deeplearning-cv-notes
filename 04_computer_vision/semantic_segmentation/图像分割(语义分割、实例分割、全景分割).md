### 基于监督下的图像分割

目前图像分割任务发展出了以下几个子领域：语义分割（semantic segmentation）、实例分割（instance segmentation）以及今年刚兴起的新领域全景分割（panoptic segmentation）。

而想要理清三个子领域的区别就不得不提到关于图像分割中 things 和 stuff 的区别：图像中的内容可以按照是否有固定形状分为 things 类别和 stuff 类别，其中，人，车等有固定形状的物体属于 things 类别（可数名词通常属于 things）；天空，草地等没有固定形状的物体属于 stuff 类别（不可数名词属于 stuff）。

语义分割更注重「类别之间的区分」，而实例分割更注重「个体之间的区分」，以下图为例，从上到下分别是原图、语义分割结果和实例分割结果。语义分割会重点将前景里的人群和背景里树木、天空和草地分割开，但是它不区分人群的单独个体，如图中的人全部标记为红色，导致右边黄色框中的人无法辨别是一个人还是不同的人；而实例分割会重点将人群里的每一个人分割开，但是不在乎草地、树木和天空的分割。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181225103541.png)

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181225103551.png)

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181225103557.png)

全景分割可以说是语义分割和实例分割的结合，下图是同一张原图的全景分割结果，每个 stuff 类别与 things 类别都被分割开，可以看到，things 类别的不同个体也被彼此分割开了。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181225103619.png)

更多：[全景分割这一年，端到端之路](http://www.zhuanzhi.ai/document/766c132ea8191a4475134fc772a9cf19)

资料&干货：

- [语义分割相关资料总结](https://zhuanlan.zhihu.com/p/41976717)

  > 看完里面的资料后，基本对语义分割可以蛮清楚了。

- 知乎：[当前主流的图像分割研究方向都有哪些？](https://www.zhihu.com/question/33599013)

综述类：

- [语义分割 | 发展综述](https://zhuanlan.zhihu.com/p/37618829)
- [十分钟看懂图像语义分割技术](https://www.leiphone.com/news/201705/YbRHBVIjhqVBP0X5.html)

  > 这篇文章介绍了“Normalized cut”的图划分方法，简称“N-cut”，还介绍到 Grab Cut，以及神经网络、FCN、条件随机场 CRF 等内容。（一篇很全面且易懂的好文）

- 知乎_魏秀参：[从特斯拉到计算机视觉之「图像语义分割」](https://zhuanlan.zhihu.com/p/21824299)

  > 看完上面那篇再看这篇吧，内容大部分差不多，但值得看看。

- [资源 | 从全连接层到大型卷积核：深度学习语义分割全指南](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650728920&idx=4&sn=3c51fa0a95742d37222c3e16b77267ca&scene=21#wechat_redirect)
- [一文概览主要语义分割网络：FCN,SegNet,U-Net...](https://www.tinymind.cn/articles/410)
- [分割算法——可以分割一切目标（各种分割总结）](https://mp.weixin.qq.com/s/KcVKKsAyz-eVsyWR0Y812A)



---

传统分割方法：

- [图像分割技术（1）](https://blog.csdn.net/zizi7/article/details/50950494)

  > 介绍了一些传统的图像分割技术。一般有 4 种分割思路：基于点线、边缘的分割；基于阈值的分割（二值化）；基于区域的分割；基于分水岭的分割

语义分割专栏：

- 知乎_stone：[图像语义分割](https://zhuanlan.zhihu.com/c_197474183)
- 知乎_学海无涯乐为舟：[语义分割的学习](https://zhuanlan.zhihu.com/c_1008415414103203840)
- 知乎_加油可好：[语义分割刷怪进阶](https://zhuanlan.zhihu.com/c_156519173)
- 专知：[图像分割](http://www.zhuanzhi.ai/topic/2001388508271825/awesome) 

语义分割相关研究：

- [图像语义分割之FCN和CRF](https://blog.csdn.net/u012759136/article/details/52434826#t9)


语义分割模型讲解：

- [全卷积网络 FCN 详解](https://zhuanlan.zhihu.com/p/30195134)

论文汇总：

- [语义分割 - Semantic Segmentation Papers](https://www.aiuai.cn/aifarm62.html)



#### 实例分割

相较于语义分割，实例分割不仅要做出像素级别的分类，还要在此基础上将同一类别不同个体分出来，即做到每个实例的分割。这对分割算法提出了更高的要求。好在我们此前积累足够的目标检测算法基础。实例分割的基本思路就是在语义分割的基础上加上目标检测，先用目标检测算法将图像中的实例进行定位，再用语义分割方法对不同定位框中的目标物体进行标记，从而达到实例分割的目的。

实例分割算法也有一定的发展历史但其中影响深远且地位重要的算法不多，以 Mask R-CNN 为例进行介绍。

Mask R-CNN 是 2017 年何恺明等大佬基于此前的两阶段目标检测算法推出的顶级网络。Mask R-CNN 的整体架构如图所示：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181031203911.png)

Mask R-CNN 将 Fast R-CNN 的 ROI Pooling 层升级成了 ROI Align 层，并且在边界框识别的基础上添加了分支 FCN 层，即 mask 层，用于语义 Mask 识别，通过 RPN 网络生成目标候选框，然后对每个目标候选框分类判断和边框回归，同时利用全卷积网络对每个目标候选框预测分割。Mask R-CNN 本质上一个实例分割算法（Instance Segmentation），相较于语义分割（Semantic Segmentation），实例分割对同类物体有着更为精细的分割。

Mask R-CNN 在 COCO 测试集上的图像分割效果如下：![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181031204014.png)

*Mask R-CNN 论文：https://arxiv.org/abs/1703.06870*

*MaskR-CNN 开源实现参考：https://github.com/matterport/Mask_RCNN*



### 基于弱监督学习的图像分割

最近基于深度学习的图像分割技术一般依赖于卷积神经网络 CNN 的训练，训练过程中需要非常大量的标记图像，即一般要求训练图像中都要有精确的分割结果。

对于图像分割而言，要得到大量的完整标记过的图像非常困难，比如在 ImageNet 数据集上，有 1400 万张图有类别标记，有 50 万张图给出了 bounding box，但是只有 4460 张图像有像素级别的分割结果。对训练图像中的每个像素做标记非常耗时，特别是对医学图像而言，完成对一个三维的 CT 或者 MRI 图像中各组织的标记过程需要数小时。如果学习算法能通过对一些初略标记过的数据集的学习就能完成好的分割结果，那么对训练数据的标记过程就很简单，这可以大大降低花在训练数据标记上的时间。这些初略标记可以是：

1. 只给出一张图像里面包含哪些物体，
2. 给出某个物体的边界框，
3. 对图像中的物体区域做部分像素的标记，例如画一些线条、涂鸦等（scribbles)。



参考文章：

- [CNN在基于弱监督学习的图像分割中的应用](https://zhuanlan.zhihu.com/p/23811946)
- [一文概览主要语义分割网络：FCN,SegNet,U-Net...](https://www.tinymind.cn/articles/410)



### 深度学习图像标注工具

- [深度学习（目标检测。图像分割等）图像标注工具汇总](https://blog.csdn.net/u012426298/article/details/80519158#t6)



### 最新进展

专知：[图像分割板块](http://www.zhuanzhi.ai/timeline) 



### 数据集介绍

(1) PASCAL VOC 数据集：

> PASCAL VOC 挑战赛是视觉对象的分类识别和检测的一个基准测试，提供了检测算法和学习性能的标准图像注释数据集和标准的评估系统。PASCAL VOC 图片集包括 20 个目录：人类；动物（鸟、猫、牛、狗、马、羊）；交通工具（飞机、自行车、船、公共汽车、小轿车、摩托车、火车）；室内（瓶子、椅子、餐桌、盆栽植物、沙发、电视）。
>
> PASCAL VOC 挑战赛在 2012 年后便不再举办，但其数据集图像质量好，标注完备，非常适合用来测试算法性能。
>
> 数据集大小：~2GB

- [PASCAL VOC数据集分析](https://blog.csdn.net/zhangjunbob/article/details/52769381)
- [深度学习图像分割（一）——PASCAL-VOC2012数据集（vocdevkit、Vocbenchmark_release）详细介绍](https://oldpan.me/archives/pascal-voc2012-guide)

(2) PASCAL-Context：

> PASCAL-Context 数据集(2014)是 PASCAL VOC 数据集(2010)的扩展。它包括了 10k 张训练图片，10k 张验证图片，以及 10k 张测试图片。新版数据集的特别之处在于整个情景被分成超过 400 个分类。注意，这些图像由 6 名内部标注师花了六个月标注完成。
>
> PASCAL-Context 官方评估标准仍然是 mloU。也有少数研究者在发表的时候采用像素准确度 (pixAcc)做为评估测度。

(3) MS COCO 数据集：

> COCO 数据集是大规模物体检测（detection）、分割（segmentation）和图说（captioning）数据集，包括 330K 图像（其中超过 200K 有注释），150 万图像实例，80 个物体类别，91 种物质（stuff）类别，每幅图有 5 条图说，250000 带有关键点的人体。
>
> COCO 数据集由微软赞助，其对于图像的标注信息不仅有类别、位置信息，还有对图像的语义文本描述，COCO 数据集的开源使得近两三年来图像分割语义理解取得了巨大的进展，也几乎成为了图像语义理解算法性能评价的“标准”数据集。Google 的开源 show and tell 生成模型就是在此数据集上测试的。 目前包含的比赛项目有：
>
> 1. 目标检测
> 2. 图像标注
> 3. 人体关键点检测
>
> 数据集大小：~40GB
>
> 注：MS COCO（Microsoft Common Objects in Context，常见物体图像识别）竞赛是继 ImageNet 竞赛（已停办）后，计算机视觉领域最受关注和最权威的比赛之一，是图像（物体）识别方向最重要的标杆（没有之一），也是目前国际领域唯一能够汇集谷歌、微软、Facebook 三大巨头以及国际顶尖院校共同参与的大赛。

(4) Cityscapes：

> Cityscapes 数据集已于 2016 年发布，包含来自 50 个城市的复杂的城市场景分割图。 它由 23.5k 张图像组成，用于训练和验证（详细和粗略的注释）和 1.5 个图像用于测试（仅详细注释）。 图像是完全分割的，例如具有 29 个类别的PASCAL-Context数据集（在 8 个超级类别中：平面，人类，车辆，建筑，物体，自然，天空，虚空）。 由于其复杂性，它通常用于评估语义分割模型。 它也因其与自动驾驶应用中的真实城市场景相似而众所周知。 诸如 PASCAL 数据集使用 mIoU 度量来评估语义分割模型的性能。

### 图像分割代码

#### FCN

GitHub 搜索：https://github.com/search?q=fcn

- [EternityZY/FCN-TensorFlow](https://github.com/EternityZY/FCN-TensorFlow)  [学习]

  > 代码讲解：[全卷积神经网络FCN-TensorFlow代码精析](https://blog.csdn.net/qq_16761599/article/details/80069824)

- 



#### SegNet

GitHub 搜索：https://github.com/search?q=segnet



####  UNet

GitHub 搜索：https://github.com/search?q=unet





#### DeepLab

GitHub 搜索：https://github.com/search?q=deeplab

- 官方代码：[tensorflow/models/research/deeplab/](https://github.com/tensorflow/models/tree/master/research/deeplab)

  > [Tensoflow-代码实战篇--Deeplab-V3+--代码复现（二）](https://blog.csdn.net/qq_38437505/article/details/83039882)

- [DrSleep/tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet)

  > This is an (re-)implementation of [DeepLab-ResNet](http://liangchiehchen.com/projects/DeepLabv2_resnet.html) in TensorFlow for semantic image segmentation on the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).
  >
  > 代码讲解：https://blog.csdn.net/wangdongwei0/article/details/82926733

- [sthalles/deeplab_v3](https://github.com/sthalles/deeplab_v3)  【准备实践该代码】

  > Tensorflow Implementation of the Semantic Segmentation DeepLab_V3 CNN
  >
  > For a complete documentation of this implementation, check out the [blog post](https://sthalles.github.io/deep_segmentation_network/). 
  >
  > 中文：[语义分割网络DeepLab-v3的架构设计思想和TensorFlow实现](https://www.jiqizhixin.com/articles/deeplab-v3)

- [anxiangSir/deeplab_v3](https://github.com/anxiangSir/deeplab_v3)



