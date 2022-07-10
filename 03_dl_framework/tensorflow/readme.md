# tensorflow 学习

## 1. tensorflow1.x 学习

### 1.1 快速入门

- [令人困惑的TensorFlow！](https://zhuanlan.zhihu.com/p/38812133)
- [令人困惑的 TensorFlow！(II)](https://zhuanlan.zhihu.com/p/46008208)

### 1.2 学习笔记

《深度学习框架Tensorflow学习与应用》笔记索引（其中会有补充一些内容）：

- [01-Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装](./《深度学习框架Tensorflow学习与应用》笔记/notes/01-Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装.md)

- [02-Tensorflow的基础使用，包括对图(graphs),会话(session),张量(tensor),变量(Variable)的一些解释和操作](./《深度学习框架Tensorflow学习与应用》笔记/notes/02-Tensorflow的基础使用，包括对图\(graphs\),会话\(session\),张量\(tensor\),变量\(Variable\)的一些解释和操作.md)

- [03-Tensorflow线性回归以及分类的简单使用](./《深度学习框架Tensorflow学习与应用》笔记/notes/03-Tensorflow线性回归以及分类的简单使用.md)

  ``` xml
  - 开始以手写数字识别 MNIST 例子来讲解，关于 MNIST 的内容还可以看看该 README 下面的
  ```

- [04-softmax，交叉熵(cross-entropy)，dropout以及Tensorflow中各种优化器的介绍](./《深度学习框架Tensorflow学习与应用》笔记/notes/04-softmax，交叉熵\(cross-entropy\)，dropout以及Tensorflow中各种优化器的介绍.md) - 

  ``` xml
  - softmax、损失函数、dropout
  - tensorflow 中各种优化器
  ```

  注：在（三）节开始的代码`4-1交叉熵.py`，发现 tf.nn.softmax_cross_entropy_with_logits 用法的小问题，[详见-传送](./Notes/tf.nn.softmax_cross_entropy_with_logits的用法问题.md)

- [05-使用Tensorboard进行结构可视化，以及网络运算过程可视化](./《深度学习框架Tensorflow学习与应用》笔记/notes/05-使用Tensorboard进行结构可视化，以及网络运算过程可视化.md)

  ``` xml
  - 用例子演示如何使结构的可视化
  - 参数细节的可视化，绘制各个参数变化情况
  - 补充内容：可视化工具 TensorBoard 更多使用和细节
  ```

- [06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题](./《深度学习框架Tensorflow学习与应用》笔记/notes/06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题.md)

  ``` xml
  - 卷积神经网络 CNN（包括局部感受野、权值共享、卷积、二维池化、多通道池化等）
  - 补充内容：参数数量的计算（以 LeNet-5 为例子）
  - 补充内容：TensorFlow 中的 Padding 到底是怎样的？ 
  - 补充内容：TensorFlow 中的卷积和池化 API 详解
  - 补充内容：TensorFlow 中的 Summary 的用法
  ```

- [07-递归神经网络LSTM的讲解，以及LSTM网络的使用](./《深度学习框架Tensorflow学习与应用》笔记/notes/07-递归神经网络LSTM的讲解，以及LSTM网络的使用.md)

- [08-保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别](./《深度学习框架Tensorflow学习与应用》笔记/notes/08-保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别.md)

  ``` xml
  - 保存模型、加载模型
  - 使用 Inception-v3 网络模型进行图像识别
  - 补充内容：加载预训练模型和保存模型以及 fine-tuning
  - 补充内容：迁移学习
  ```

- [09-Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别](./《深度学习框架Tensorflow学习与应用》笔记/notes/09-Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别.md)

  ``` xml
  - TensorFlow 的 GPU 版本安装
  - 使用 inception-v3 模型进行训练预测
  - 使用 tensorflow 已经训练好的模型进行微调
  - 制作 `.tfrecord` 存储文件
  ```

- [10-使用Tensorflow进行验证码识别](./《深度学习框架Tensorflow学习与应用》笔记/notes/10-使用Tensorflow进行验证码识别.md)

- [11-Tensorflow在NLP中的使用(一)](./《深度学习框架Tensorflow学习与应用》笔记/notes/11-Tensorflow在NLP中的使用\(一\).md)

- [12-Tensorflow在NLP中的使用(二)](./《深度学习框架Tensorflow学习与应用》笔记/notes/12-Tensorflow在NLP中的使用\(二\).md)

笔记补充：

- 对 TensorFlow 的再次理解和总结：[TensorFlow的理解和总结](./[转]TensorFlow的理解和总结.md)
- 对 TensorFlow 的 API 使用记录下来，方便查阅：🔎 [TensorFlow的API详解和记录](./[整理]TensorFlow的API详解和记录.md) ★★★ 【荐】
- TensorFlow 使用指定的 GPU 以及显存分析：[tensorflow中使用指定的GPU及显存分析](./tensorflow中使用指定的GPU及显存分析.md)  【荐】

### 1.3 学习来源

学习来源：炼数成金的《深度学习框架TensorFlow学习与应用》视频 + 网上博客内容  。

视频目录：

```xml
第 1周 Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装
第 2周 Tensorflow的基础使用，包括对图(graphs),会话(session),张量(tensor),变量(Variable)的一些解释和操作
第 3周 Tensorflow线性回归以及分类的简单使用
第 4周 softmax，交叉熵(cross-entropy)，dropout以及Tensorflow中各种优化器的介绍
第 5周 卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题
第 6周 使用Tensorboard进行结构可视化，以及网络运算过程可视化
第 7周 递归神经网络LSTM的讲解，以及LSTM网络的使用
第 8周 保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别
第 9周 Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别
第10周 使用Tensorflow进行验证码识别
第11周 Tensorflow在NLP中的使用(一)
第12周 Tensorflow在NLP中的使用(二)
```

> 说明：实际第 5 周讲的是 tensorborad 结构可视化，第 6 周讲的是 CNN，下面网盘该视频的文件夹顺序，我已修正。

在线观看：

- YouTube：[tensorflow教程（十课）](https://www.youtube.com/watch?v=eAtGqz8ytOI&list=PLjSwXXbVlK6IHzhLOMpwHHLjYmINRstrk&index=2&t=0s)
- 或 B 站：[《深度学习框架TensorFlow学习与应用》](https://www.bilibili.com/video/av20542427/)

视频下载：

- 《深度学习框架Tensorflow学习与应用》（含视频+代码+课件，视频总时长：13小时31分钟）  【[百度网盘下载](https://pan.baidu.com/s/16OINOrFiRXbqmqOFjCFzLQ )  密码: 1a8j】
- 《深度学习框架Tensorflow学习与应用[只有videos-720p]》（该份资料只有视频文件） 【 [百度网盘下载](https://pan.baidu.com/s/1oQLgWFEBsVrcKJN4swEdzg)  密码: i3e2】

### 1.4 其他资料

其他学习视频，觉得有必要可以看看：

- 油管视频：[TF Girls 修炼指南](https://www.youtube.com/watch?v=TrWqRMJZU8A&list=PLwY2GJhAPWRcZxxVFpNhhfivuW0kX15yG&index=2)  或 B 站观看： [TF Girls 修炼指南](https://space.bilibili.com/16696495/#/channel/detail?cid=1588) 
- 油管视频：51CTO视频 [深度学习框架-Tensorflow案例实战视频课程](https://www.youtube.com/watch?v=-pYU4ub7g0c&list=PL8LR_PrSuIRhpEYA3sJ-J5hYGYUSwZwdS)、或 B 站观看：[深度学习框架-Tensorflow案例实战视频课程](https://www.bilibili.com/video/av29663946/?p=1)
- [Tensorflow 教程系列 | 莫烦Python](<https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/>)

相关资料：

- 郑泽宇/顾思宇：[《Tensorflow：实战Google深度学习框架》](https://book.douban.com/subject/26976457/) 出版时间 2017-2-10
  - 官方维护的书中的 TensorFlow 不同版本的示例程序仓库：<https://github.com/caicloud/tensorflow-tutorial>；
  - 有人在 GitHub 上写了笔记：[TensorFlow_learning_notes](https://github.com/cookeem/TensorFlow_learning_notes)
- 黄文坚/唐源：[《TensorFlow实战》](https://book.douban.com/subject/26974266/) 出版时间 2017-2-1
  - 源码实现：<https://github.com/terrytangyuan/tensorflow-in-practice-code>
- 掘金翻译：[TensorFlow 最新官方文档中文版 V1.10 ](https://github.com/xitu/tensorflow-docs)
- 极客学院：[TensorFlow 官方文档中文版](http://wiki.jikexueyuan.com/project/tensorflow-zh/)
- [TensorFlow 官方文档中文版](http://www.tensorfly.cn/tfdoc/get_started/introduction.html)

## 2. tensorflow2.x 学习

- [czy36mengfei/tensorflow2_tutorials_chinese](<https://github.com/czy36mengfei/tensorflow2_tutorials_chinese>) - tensorflow2中文教程，持续更新(当前版本:tensorflow2.0)

  
