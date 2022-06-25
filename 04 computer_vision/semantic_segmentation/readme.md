#### (1) 图像分割基础

①什么是图像分割？

- [图像分割 传统方法 整理](https://zhuanlan.zhihu.com/p/30732385)  [荐看完]

  图片分割根据灰度、颜色、纹理、和形状等特征将图像进行划分区域，让区域间显差异性，区域内呈相似性。主要分割方法有：

  ``` xml
  基于阈值的分割
  基于边缘的分割
  基于区域的分割
  基于图论的分割
  基于能量泛函的分割
  ```

- [十分钟看懂图像语义分割技术 | 雷锋网](https://www.leiphone.com/news/201705/YbRHBVIjhqVBP0X5.html)  [荐看完]

②综述类/总结类：

- [从全连接层到大型卷积核：深度学习语义分割全指南](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650728920&idx=4&sn=3c51fa0a95742d37222c3e16b77267ca&scene=21#wechat_redirect)

- [分割算法——可以分割一切目标（各种分割总结）](https://mp.weixin.qq.com/s/KcVKKsAyz-eVsyWR0Y812A)  [荐]

  深度学习最初流行的分割方法是，打补丁式的分类方法 (patch classification) 。逐像素地抽取周围像素对中心像素进行分类。由于当时的卷积网络末端都使用全连接层 (full connected layers) ，所以只能使用这种逐像素的分割方法。

  但是到了 2014 年，来自伯克利的 Fully Convolutional Networks（FCN）卷积网络，去掉了末端的全连接层。随后的语义分割模型基本上都采用了这种结构。除了全连接层，语义分割另一个重要的问题是池化层。池化层能进一步提取抽象特征增加感受域，但是丢弃了像素的位置信息。但是语义分割需要类别标签和原图像对齐，因此需要从新引入像素的位置信息。有两种不同的架构可以解决此像素定位问题。

  第一种是`编码-译码架构`。编码过程通过池化层逐渐减少位置信息、抽取抽象特征；译码过程逐渐恢复位置信息。一般译码与编码间有直接的连接。该类架构中 U-net 是最流行的。

  第二种是`膨胀卷积` (dilated convolutions) 【这个核心技术值得去阅读学习】，抛弃了池化层。

- [一文概览主要语义分割网络：FCN,SegNet,U-Net...](https://www.tinymind.cn/articles/410)

  该文为译文，介绍了很多语义分割的深度学习模型，包括半监督下的语义分割，可以大致看下。

③深度学习语义分割模型的介绍：

- [语义分割(semantic segmentation) 常用神经网络介绍对比-FCN SegNet U-net DeconvNet](https://blog.csdn.net/zhyj3038/article/details/71195262)
- [深度学习（十九）——FCN, SegNet, DeconvNet, DeepLab, ENet, GCN](https://blog.csdn.net/antkillerfarm/article/details/79524417)

④图像分割的衡量指标：

- [图像分割的衡量指标详解](https://blog.csdn.net/qq_37274615/article/details/78957962)

语义分割其实就是对图片的每个像素都做分类。其中，较为重要的语义分割数据集有：VOC2012 以及 MSCOCO。

#### (2) 图像分割仓库

- [semseg](https://github.com/guanfuchen/semseg)

  > 常用的语义分割架构结构综述以及代码复现

- [DeepNetModel](https://github.com/guanfuchen/DeepNetModel)

  > 记录每一个常用的深度模型结构的特点（图和代码）
  >
  > 大佬的博客：[计算机视觉相关资源整理](https://guanfuchen.github.io/post/markdown_blog_ws/markdown_blog_2017_11/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E7%9B%B8%E5%85%B3%E8%B5%84%E6%BA%90%E6%95%B4%E7%90%86/)

- [Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)

  > Semantic Segmentation Suite in TensorFlow. Implement, train, and test new Semantic Segmentation models easily!

- [mrgloom/awesome-semantic-segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)（图像分割论文下载及实现可以在这里找到~）

#### (3) 图像分割论文及最新研究

论文汇集：

- [语义分割 - Semantic Segmentation Papers](https://blog.csdn.net/zziahgf/article/details/72639791)



#### (4) 图像分割讲解视频

- [浙大博士生刘汉唐：带你回顾图像分割的经典算法](http://www.mooc.ai/course/414/learn#lesson/2266)（需要注册才能观看~）
- [197期\_张觅\_基于深度卷积网络的遥感影像语义分割层次认知方法](https://www.bilibili.com/video/av24599502?from=search&seid=11210211322309323243)（关于遥感图像语义分割的，但听得不是很清楚~）
- [【 计算机视觉 】深度学习语义分割Semantic Segmentation（英文字幕）（合辑）_哔哩哔哩](<https://www.bilibili.com/video/av21286423/?p=1>)