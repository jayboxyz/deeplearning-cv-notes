# （1）

- Rectified Linear Unit(ReLU) - 用于隐层神经元输出
- Sigmoid - 用于隐层神经元输出
- Softmax - 用于多分类神经网络输出
- Linear - 用于回归神经网络输出（或二分类问题）

Softmax 激活函数只用于多于一个输出的神经元，它保证所以的输出神经元之和为1.0，所以一般输出的是小于1的概率值，可以很直观地比较各输出值。



## 为什么选择ReLU？

深度学习中，我们一般使用 ReLU 作为中间隐层神经元的激活函数，AlexNet 中提出用 ReLU 来替代传统的激活函数是深度学习的一大进步。我们知道，sigmoid 函数的图像如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190424093949.png)

而一般我们优化参数时会用到误差反向传播算法，即要对激活函数求导，得到sigmoid函数的瞬时变化率，其导数表达式为：![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190424094005.png)

对应的图形如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190424094019.png)

由图可知，导数从0开始很快就又趋近于0了，易造成“梯度消失”现象，而ReLU的导数就不存在这样的问题，它的导数表达式如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190424094039.png)

Relu函数的形状如下（蓝色）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190424094056.png)

对比sigmoid类函数主要变化是：1）单侧抑制 2）相对宽阔的兴奋边界 3）稀疏激活性。这与人的神经皮层的工作原理接近。

## 为什么需要偏移常量？

通常，要将输入的参数通过神经元后映射到一个新的空间中，我们需要对其进行加权和偏移处理后再激活，而不仅仅是上面讨论激活函数那样，仅对输入本身进行激活操作。比如sigmoid激活神经网络的表达式如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190424094129.png)

x是输入量，w是权重，b是偏移量（bias）。这里，之所以会讨论sigmoid函数是因为它能够很好地说明偏移量的作用。

权重w使得sigmoid函数可以调整其倾斜程度，下面这幅图是当权重变化时，sigmoid函数图形的变化情况：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190424094158.png)

 上面的曲线是由下面这几组参数产生的：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190424094229.png)

我们没有使用偏移量b（b=0），从图中可以看出，无论权重如何变化，曲线都要经过（0,0.5）点，但实际情况下，我们可能需要在x接近0时，函数结果为其他值。下面我们改变偏移量b，它不会改变曲线大体形状，但是改变了数值结果：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190424094242.png)

上面几个sigmoid曲线对应的参数组为：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190424094254.png)

这里，我们规定权重为 1，而偏移量是变化的，可以看出它们向左或者向右移动了，但又在左下和右上部位趋于一致。

当我们改变权重 w 和偏移量 b 时，可以为神经元构造多种输出可能性，这还仅仅是一个神经元，在神经网络中，千千万万个神经元结合就能产生复杂的输出模式。

参考：

- [深度学习常用激活函数之— Sigmoid & ReLU & Softmax](https://blog.csdn.net/Leo_Xu06/article/details/53708647)
- [交叉熵代价函数（cross-entropy cost function）](<https://blog.csdn.net/wtq1993/article/details/51741471>)

#（2）

> 只是映射了之后，相对大小没发生变化。该大的还是大，该小的还是小。人们为了方便当做概率罢了。
>
> ——from：通过逻辑回归的sigmoid函数把线性回归转化到[0,1]之间，这个值为什么可以代表概率？ - zkjjj的回答 - 知乎
> https://www.zhihu.com/question/41647192/answer/221104515

> 怎么说呢，就拿一个简单LR二分类举例吧，识别是否为西瓜。
> 我们拿掉sigmoid，结果输出为一个值，例如100，然后模型训练结果是大于60就属于西瓜，没问题。
> 现在加上sigmoid，整个输出值的范围从之前的正负无穷压缩到0到1，因为我们不能取到0和1，只是不断趋近，那么越接近1就说明原来的值越接近无限大，也就是离分割线（超平面）越远，那么属于正例的可能自然无限大了。
> 简单来说，sigmoid只是把可能性大小压缩到0到100%之内，方便计算而已。
>
> ——from：https://www.zhihu.com/question/41647192/answer/216501244



本质上来说，Softmax 属于离散概率分布而 Sigmoid 是非线性映射。分类其实就是设定一个阈值，然后我们将想要分类的对象与这个阈值进行比较，根据比较结果来决定分类。**Softmax 函数能够将一个K维实值向量归一化**，所以它主要被用于多分类任务；**Sigmoid 能够将一个实数归一化**，因此它一般用于二分类任务。特别地，当 Softmax 的维数 K=2  时，Softmax 会退化为 Sigmoid 函数。——from：[快速理解Softmax和Sigmoid]([https://lolimay.cn/2019/01/14/%E5%BF%AB%E9%80%9F%E7%90%86%E8%A7%A3-Softmax-%E5%92%8C-Sigmoid/](https://lolimay.cn/2019/01/14/快速理解-Softmax-和-Sigmoid/))