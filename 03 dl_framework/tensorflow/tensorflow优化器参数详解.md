## learning rate、weight decay和momentum等

### 超参数

所谓**超参数**，就是机器学习模型里面的框架参数，比如聚类方法里面类的个数，或者话题模型里面话题的个数等等，都称为超参数。它们跟训练过程中学习的参数（权重）是不一样的，通常是手工设定，不断试错调整，或者对一系列穷举出来的参数组合一通枚举（叫做网格搜索）。深度学习和神经网络模型，有很多这样的参数需要学习，这就是为什么过去这么多年从业者弃之不顾的原因。以前给人的印象，深度学习就是“黑魔法”。时至今日，非参数学习研究正在帮助深度学习更加自动的优化模型参数选择，当然有经验的专家仍然是必须的。

说到这些参数就会想到**Stochastic Gradient Descent (SGD)！其实**这些参数在**caffe.proto**中 对caffe网络中出现的各项参数做了详细的解释。



### Learning Rate

**学习率决定了权值更新的速度**，设置得**太大会使结果超过最优值**，**太小会使下降速度过慢**。仅靠人为干预调整参数需要不断修改学习率，因此后面 3 种参数都是基于自适应的思路提出的解决方案。后面 3 中参数分别为：Weight Decay 权值衰减，Momentum 动量和 Learning Rate Decay 学习率衰减。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429165258.png)



### Weight decay

在实际应用中，为了避免网络的过拟合，必须对价值函数（Cost function）加入一些正则项，在SGD中加入![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429165531.png)这一正则项对这个Cost function进行规范化：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429165556.png)

上面这个公式基本思想就是减小不重要的参数对最后结果的影响，网络中有用的权重则不会收到Weight decay影响。

**在机器学习或者模式识别中**，会出现 overfitting，而当网络逐渐 overfitting 时网络权值逐渐变大，因此，为了避免出现 overfitting，会给误差函数添加一个惩罚项，常用的惩罚项是所有权重的平方乘以一个衰减常量之和。**其用来惩罚大的权值。**

weight decay（权值衰减）的使用**既不是为了提高收敛精确度也不是为了提高收敛速度**，其最终目的是**防止过拟合**。在损失函数中，weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。

momentum是梯度下降法中一种常用的加速技术。对于一般的SGD，其表达式为, x 沿负梯度方向下降。而带momentum项的SGD则写生如下形式：

。。。



### Momentum

动量来源于牛顿定律，基本思想是为了找到最优加入“惯性”的影响，当误差曲面中存在平坦区域，SGD 就可以更快的学习。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429165629.png)

### **Learning Rate Decay** 

该方法是为了提高SGD寻优能力，具体就是每次迭代的时候减少学习率的大小。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429165647.png)



---

---



### 学习速率（learning rate，η）

运用梯度下降算法进行优化时，权重的更新规则中，在梯度项前会乘以一个系数，这个系数就叫学习速率η。下面讨论在训练时选取η的策略。

- 固定的学习速率。如果学习速率太小，则会使收敛过慢，如果学习速率太大，则会导致代价函数振荡，如下图所示。就下图来说，一个比较好的策略是先将学习速率设置为0.25，然后在训练到第20个Epoch时，学习速率改为0.025。

关于为什么学习速率太大时会振荡，看看这张图就知道了，绿色的球和箭头代表当前所处的位置，以及梯度的方向，学习速率越大，那么往箭头方向前进得越多，如果太大则会导致直接跨过谷底到达另一端，所谓“步子太大，迈过山谷”。

在实践中，怎么粗略地确定一个比较好的学习速率呢？好像也只能通过尝试。你可以先把学习速率设置为0.01，然后观察training cost的走向，如果cost在减小，那你可以逐步地调大学习速率，试试0.1，1.0….如果cost在增大，那就得减小学习速率，试试0.001，0.0001….经过一番尝试之后，你可以大概确定学习速率的合适的值。

**为什么是根据training cost来确定学习速率，而不是根据validation accuracy来确定呢？这里直接引用一段话，有兴趣可以看看：**

``` xml
This all seems quite straightforward. However, using the training cost to pick η appears to contradict what I said earlier in this section, namely, that we’d pick hyper-parameters by evaluating performance using our held-out validation data. In fact, we’ll use validation accuracy to pick the regularization hyper-parameter, the mini-batch size, and network parameters such as the number of layers and hidden neurons, and so on. Why do things differently for the learning rate? Frankly, this choice is my personal aesthetic preference, and is perhaps somewhat idiosyncratic. The reasoning is that the other hyper-parameters are intended to improve the final classification accuracy on the test set, and so it makes sense to select them on the basis of validation accuracy. However, the learning rate is only incidentally meant to impact the final classification accuracy. It’s primary purpose is really to control the step size in gradient descent, and monitoring the training cost is the best way to detect if the step size is too big. With that said, this is a personal aesthetic preference. Early on during learning the training cost usually only decreases if the validation accuracy improves, and so in practice it’s unlikely to make much difference which criterion you use.
```

### Early Stopping

所谓early stopping，即在每一个epoch结束时（一个epoch即对所有训练数据的一轮遍历）计算 validation data的accuracy，当accuracy不再提高时，就停止训练。这是很自然的做法，因为accuracy不再提高了，训练下去也没用。另外，这样做还能防止overfitting。

那么，怎么样才算是validation accuracy不再提高呢？并不是说validation accuracy一降下来，它就是“不再提高”，因为可能经过这个epoch后，accuracy降低了，但是随后的epoch又让accuracy升上去了，所以不能根据一两次的连续降低就判断“不再提高”。正确的做法是，在训练的过程中，记录最佳的validation accuracy，当连续10次epoch（或者更多次）没达到最佳accuracy时，你可以认为“不再提高”，此时使用early stopping。这个策略就叫“ no-improvement-in-n”，n即epoch的次数，可以根据实际情况取10、20、30….

### 可变的学习速率

在前面我们讲了怎么寻找比较好的learning rate，方法就是不断尝试。在一开始的时候，我们可以将其设大一点，这样就可以使weights快一点发生改变，从而让你看出cost曲线的走向（上升or下降），进一步地你就可以决定增大还是减小learning rate。

但是问题是，找出这个合适的learning rate之后，我们前面的做法是在训练这个网络的整个过程都使用这个learning rate。这显然不是好的方法，在优化的过程中，learning rate应该是逐步减小的，越接近“山谷”的时候，迈的“步伐”应该越小。

在讲前面那张cost曲线图时，我们说可以先将learning rate设置为0.25，到了第20个epoch时候设置为0.025。**这是人工的调节，而且是在画出那张cost曲线图之后做出的决策。能不能让程序在训练过程中自动地决定在哪个时候减小learning rate？**

答案是肯定的，而且做法很多。一个简单有效的做法就是，当validation accuracy满足 no-improvement-in-n规则时，本来我们是要early stopping的，但是我们可以不stop，而是让learning rate减半，之后让程序继续跑。下一次validation accuracy又满足no-improvement-in-n规则时，我们同样再将learning rate减半（此时变为原始learni rate的四分之一）…继续这个过程，直到learning rate变为原来的1/1024再终止程序。（1/1024还是1/512还是其他可以根据实际确定）。【PS：也可以选择每一次将learning rate除以10，而不是除以2.】

### 正则项系数（regularization parameter, λ）

正则项系数初始值应该设置为多少，好像也没有一个比较好的准则。建议一开始将正则项系数λ设置为0，先确定一个比较好的learning rate。然后固定该learning rate，给λ一个值（比如1.0），然后根据validation accuracy，将λ增大或者减小10倍（增减10倍是粗调节，当你确定了λ的合适的数量级后，比如λ = 0.01,再进一步地细调节，比如调节为0.02，0.03，0.009之类。）

在《Neural Networks：Tricks of the Trade》中的第三章『A Simple Trick for Estimating the Weight Decay Parameter』中，有关于如何估计权重衰减项系数的讨论，有基础的读者可以看一下。

### Mini-batch size

首先说一下采用mini-batch时的权重更新规则。比如mini-batch size设为100，则权重更新的规则为：

也就是将100个样本的梯度求均值，替代online learning方法中单个样本的梯度值：

当采用mini-batch时，我们可以将**一个batch里的所有样本放在一个矩阵里，利用线性代数库来加速梯度的计算**，这是工程实现中的一个优化方法。

那么，size要多大？一个大的batch，可以充分利用矩阵、线性代数库来进行计算的加速，batch越小，则加速效果可能越不明显。当然batch**也不是越大越好，太大了，权重的更新就会不那么频繁，导致优化过程太漫长**。所以mini-batch size选多少，不是一成不变的，根据你的数据集规模、你的设备计算能力去选。

**更多资料：**

> LeCun在1998年的论文《Efficient BackProp》
>
> Bengio在2012年的论文《Practical recommendations for gradient-based training of deep architectures》,给出了一些建议，包括梯度下降、选取超参数的详细细节。
>
> 以上两篇论文都被收录在了2012年的书《Neural Networks: Tricks of the Trade》里面，这本书里还给出了很多其他的tricks。



来源：[超参数简单理解-->learning rate,weight decay和momentum - 程序员深度学习 - CSDN博客](<https://blog.csdn.net/sinat_24143931/article/details/78863047>)

---

---

---



之前文章介绍的批量梯度下降GD就是一种优化器，通过它可以不断优化，得到最终比较优的w和b。但是我们一般不会直接使用BGD优化器，主要原因如下：

**缺点**：

1）每次迭代需要对所有训练样本进行一次操作，如果训练样本太大，训练速度会非常慢。

2）不利于增量学习，或者说在线学习，一旦有新的样本过来，整个模型需要重新训练。

**优点**：

可以比较平滑收敛到凸函数的全局最优，非凸函数可以收敛到局部最优（其实当维度很多时候，往往都是全局最优，以后会分析这个问题）

**所以梯度下降有以下形式**：

1）梯度下降（BGD）：每次要用所有样本

2）随机梯度下降（SGD）：每次一个样本，而且样本是打乱顺序的，所以用了随机这个词。

3）Batch随机梯度下降（MBGD，也叫min-batch GD）：每次多个样本，将所有样本分成多组，每组数目一样，每次迭代需要分别对每组进行学习。

4）改进后的随机梯度下降（很多变种）：学习率自动下降

（略。。。

**在介绍优化器之前，还需要介绍几个概念**：指数加权平均，momentum，decay，nesterov



### 指数加权平均

已知当前和之前的温度，如何预测未来温度？

已知温度变化：

略。。。。

通过此，我们可以知道，指数加权平均可以很好地根据历史值，估计下一个值。我们就可以用它来预测学习率！！

另外一个问题就是初始几天的值，从0开始，实际上和真实值不准，这个叫做偏差，如图，紫色线是预测值得线，红色是真实值。

通过如下公式可以得到绿色线，进行修正：

其实如果我们训练迭代次数多，可以不考虑最初的偏差，因为后面都是拟合的。

**指数加权平均优点**：

1）每次考虑前m个值，由β决定

2）前m个值越近的权值更大，越远的权值更小，最后取平均，更合理

3）修正后初始值也更符合

以上就是指数加权平均，后面很多地方会用到。

### momentum

梯度下降法可以沿梯度方向一直走，达到最优。但是如下图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429171113.png)

普通梯度下降法会沿蓝色线，不断接近最优点。

**原因**：

有很多w参数，有的w的梯度方向变化比较小，有的w梯度方向变化大，一起作用，使得其出现蓝色路径。

是否有办法让梯度变化大的w有更大权值变化，而让梯度变化小的w有更小的权值，这样就可以让其按照红色路径达到最优，更快达到最优？

**答案**：加入momentum

如何加入momentum？

（略。。。



### Nesterov accelerated gradient

其实Nesterov是在momentum基础上进行了进一步修正：

蓝色是 Momentum 的过程，会先计算当前的梯度，然后在更新后的累积梯度后会有一个大的跳跃。就是公式的两个向量相加。

而 NAG 会先在前一步的累积梯度上(brown vector)有一个大的跳跃，然后衡量一下梯度做一下修正(red vector)，这种预期的更新可以避免我们走的太快。也是新公式两项相加。



**一般框架主要的优化器有以下几种**：

- SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
- Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
- Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
- RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
- Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
- Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
- Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

其中大部分参数在上面都有介绍，这里就不一一详细介绍。

`SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)`：

实际上应该是mini-batch

**特点**：

1）lr是学习率

2）decay参数

3）momentum的β

4）nesterov是否用NAG

SGD应用momentum和NAG都是改变梯度值，而不是学习率，decay是改变学习率，但是对所有参数都是相同改变，能否让学习率对不同参数自适应更新？

比如有的参数可能已经到了仅需要微调的阶段，但又有些参数由于对应样本少等原因，还需要较大幅度的调动。

请看如下方法。

（略。。。



### 经验总结：

1）对于稀疏数据，尽量使用学习率可自适应的优化方法，不用手动调节，而且最好采用默认值：即 Adagrad, Adadelta, RMSprop, Adam，Nadam

2）SGD通常训练时间更长，但是在好的初始化和学习率调度方案的情况下，结果更可靠

3）如果在意更快的收敛，并且需要训练较深较复杂的网络时，推荐使用学习率自适应的优化方法。

4）Adadelta，RMSprop，Adam是比较相近的算法，在相似的情况下表现差不多。

5）Adam 就是在 RMSprop 的基础上加了 bias-correction 和 momentum

6）Nadam 就是Adam基础上加了NAG

7）随着梯度变的稀疏，Adam 比 RMSprop 效果会好

8）在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果

9）如果需要更快的收敛，或者是训练更深更复杂的神经网络，需要用一种自适应的算法

来源：[深度学习：原理简明教程08-深度学习:优化器 – Ling之博客]([http://www.bdpt.net/cn/2017/12/21/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%EF%BC%9A%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B08-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%BC%98%E5%8C%96%E5%99%A8/](http://www.bdpt.net/cn/2017/12/21/深度学习：原理简明教程08-深度学习优化器/))



---

---



What is Gradient Desecnt ?
---

那么，什么是梯度下降呢？学习机器学习的同学们常会遇到下面这样的图像, 这个图像看上去好复杂, 不过还挺好看的，= . =。在这张图上我们做的事情是，从某一个随机的点出发，一步一步的下降到某个最小值所在的位置。

对于人来说，我们是很容易知道那条路径下降最快的，可是机器却不是那么聪明的。他们只能靠算法，找到这个最快下降的路径。这个算法，就是我们所说的梯度下降了。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429171904.png)

其实梯度下降只是一个大的问题家族中的一员，这个家族的名字就是 - ”optimization” (优化问题)。优化能力是人类历史上的重大突破, 他解决了很多实际生活中的问题。从而渐渐演化成了一个庞大的家族.比如说牛顿法 (Newton’s method), 最小二乘法(Least Squares method), 梯度下降法 (Gradient Descent) 等等。而我们的神经网络就是属于梯度下降法这个分支中的一个。

提到梯度下降, 我们不得不说说大学里面学习过的求导求微分。因为这就是传说中”梯度下降”里面的”梯度” (gradient)啦。听到求导微分，大家不要害怕, 因为这个博客只是让你有一个直观上的理解, 并不会涉及太过复杂的数学公式推导哈。

Accelerate Gradient Desecnt
---

越复杂的神经网络, 就会需要越多的数据来训练, 导致我们在训练神经网络的过程中花费的时间也就越多。我们不想花费这么多的时间在训练上. 可是往往有时候为了解决复杂的问题, 复杂的结构和大数据又是不能避免的，这就让人很抓狂了。

所以我们（迫切的）需要寻找一些方法, 让神经网络聪明起来, 下降的速度快起来。常见的加速模式有以下几种：

- Stochastic Gradient Descent (SGD)
- Momentum
- AdaGrad
- RMSProp
- Adam

Stochastic Gradient Descent
---

随机梯度下降，应该是最基础的一种加速算法了，Gradient Descent 利用全部数据，来计算梯度下降的方向，这肯定是能找到最优的下降方向，但是同时也造成了计算量过大，时间过长的后果。

我们换一种思路, 如果把这些训练数据拆分成小批小批的, 然后再分批不断放入 NN 中计算, 这就是我们常说的 SGD 的正确打开方式了. 每次使用批数据, 虽然不能反映整体数据的情况, 不过却很大程度上加速了 NN 的训练过程, 而且也不会丢失太多准确率.

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429172058.png)

例如，上图中，如果我们是梯度下降的话，我们每一步都会走最正确的下降方向，不会走弯路，那就是图中红色线所表示的走法，如果是随机梯度下降的话，就是紫色线路所代表的走法，他可能会绕一些弯路，但是扭扭曲曲，却还是能朝着最小值的地方走过去。

注意，往往随机梯度下降，不会刚刚好就找到最小的地方，可能会在最小值附近转来转去的。

Momentum
---

如果说在训练的时候，我们觉得随机梯度下降 SGD 算法依然太慢了，怎么办呢？这时，我们就可以考虑使用 Momentum 了，Momentum 算法它是可以帮助到快速的梯度下降的。

在真正进入到算法讲解之前，我们先来大致看一下，在大多数训练的过程中，那些曲折无比的学习过程吧。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429172137.png)

例如上图中的梯度下降的例子，这个梯度下降的过程，很像是在某一个轴附近摆来摆去，虽然确实是一直朝着中间的洼地在走着，但是好像来回的摆动确实是浪费了许多步骤，看上去像一个喝醉了的人，晃晃悠悠的向着地低走去。

实际上，由于真实的训练过程中，维度远比 2 要高的多，这种无意义的摆动的情况出现的情况，出现的次数非常多。

那么 Momentum 就是发现了这种问题，对 SGD 算法做了一些改进。一句话来概括它的特点的话，那就是：计算在梯度下降的方向上，计算一个指数加权平均（ Exponentially weighted average ），利用这个来代替权重更新的方法。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429172157.png)

用通俗一点的方法来说，就是**“如果梯度下降显示，我们在一直朝着某一个方向在下降的话，我让这个方向的的学习速率快一点，如果梯度下降在某一个方向上一直是摆来摆去的，那么就让这个方向的学习速率慢一点”**。

我们可以先来看一下，Momentum 减少摆动的效果。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429172238.png)

上面两张图，就是我们没有加入，和加入了 Momentum 之后的梯度下降的变化，可以看到，有了 Momentum 之后，算法加速了横轴下降的速度，并减缓了纵轴的摆动的频率，所以在最终的训练过程中，它的步伐迈的更大，也更正确了，对比原来没有 Momentum，仅仅四步，就离最终的最小值更近了许多。

Exponentially Weighted Average
---

如果你只是想知道 Momentum 的大致原理，不想知道 Momentum 的细节的话，那么这一部分你是可以跳过的啦。

前面我们提到，Momentum 是用到**滑动平均**来获得平滑波动的效果的。那么什么是滑动平均呢？

滑动平均实际上是一种用来处理“数字序列”的方法，你们玩的股票里面的各种多少天的线，周线，月线，年线，等等，都是这种方法弄出来的。具体是怎么做的呢？假设，我们有一些带噪音的序列的序列 S。在这个例子中，我绘制了余弦函数并添加了一些高斯噪声。它看起来像这样：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429172400.png)

请注意，即使这些点看起来非常接近每个点，但它们都不共享 x 坐标。这是每个点的唯一编号。这是定义我们序列 S 中每个点的索引的数字。

我们想要对这些数据进行处理，而不是使用这些数据，我们需要某种“移动”平均值，这会使数据“降噪”并使其更接近原始功能。指数平均值可以给我们一张看起来像这样的图片：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429172444.png)

正如你所看到的那样，这是一个相当不错的结果。我们得到了更平滑的线条，而不是具有很多噪音的数据，这比我们的数据更接近原始功能。指数加权平均值用以下等式定义新的序列 V ：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429172603.png)

序列 S 是我们原始的带噪音的序列，序列 V 是我们经过滑动平均得到曲线，也就是上面绘制的黄色。 Beta 是一个从 0 到 1 的超参数。上面的 beta = 0.9。这是一个非常常用的值，最常用于带 Momentum 的 SGD。我们对序列的最后做 1 /（1- beta）进行近似平均。

让我们看看beta的选择如何影响我们的新序列 V 的吧。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429172634.png)

正如上图所示，beta 值越大，曲线与含有噪音的值越接近，beta 越小，曲线越平滑。

接下来，我们来一波数学公式推导吧。

（略。。。

那么从数学公式的角度上来说，Momentum 为什么是可以达到加速的效果的呢。请记住一点就好，Momentum 带有了滑动平均，滑动平均会将上下浮动的噪音式特征抹去，也即是我们在上图中看到的摆来摆去的动作幅度，就被滑动平均给抹平啦，但是如果你一直在朝着某一个方向走的话，滑动平均是抹不平的，Momentum 还有一个好处，就是可以冲出局部最优解，或者某一些梯度为 0 的地方，这也是因为滑动平均使他带有一定的惯性，能冲出洼地。

### 补充：滑动平均

在使用梯度下降算法训练模型时，每次更新权重时，为每个权重维护一个影子变量，该影子变量随着训练的进行，会最终稳定在一个接近真实权重的值的附近。那么，在进行预测的时候，使用影子变量的值替代真实变量的值，可以得到更好的结果。

1. 训练阶段：为每个可训练的权重维护影子变量，并随着迭代的进行更新；
2. 预测阶段：使用影子变量替代真实变量值，进行预测。

参考：[滑动平均模型在Tensorflow中的应用](https://www.jianshu.com/p/463f12f7a344)

Nesterov Accelerated Gradient
---

Nesterov Momentum 是一个稍有不同的 Momentum 更新版本，最近很流行。在这个 Nesterov 中，我们首先查看当前动量所指向的点，然后计算此时的梯度。如下图所示：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190429172837.png)

（略。。。

Which Optimizer to Use ?
---

最后的最后，我们来简单说一下，我们应该使用哪个优化器吧？如果我们的输入数据很稀少，那么可能会使用自适应学习率方法中的一种，来获得最佳结果。这样的一个好处是，我们不需要调整学习速率，它可能就会达到默认值的最佳结果。

自适应学习率方法中，RMSprop 是 AdaGrad 的延伸，它解决了其学习速度急剧下降的问题，Adam 最后为 RMSprop 增加了偏差修正和动力。就此而言，RMSprop 和 Adam 是非常相似的算法，在相似的情况下，Kingma等人表明，偏差修正有助于 Adam 在优化结束时略微优于 RMSprop ，因为梯度变得更加稀疏。就目前而言，**Adam 可能是最好的整体选择**。

一个比较有意思的事情是，看似最不好的 SGD 下降方法（因为它花费的时间最长），有时却能达到最好的学习效果。如果想要做到这一点，我们需要三个重要的地方下额外的功夫：

1. 一个非常好的初始值的设定。
2. 学习速率的 decay
3. 类似于模拟退火算法的让梯度下降不要卡在鞍部

当然，如果你不想要这么复杂的话，Adam 应该能满足你的需求了。

来源：<https://alphafan.github.io/posts/grad_desc.html>



---

---



## 个人理解：学习率（learning rate）和动量（Momentum）区别

动量（momentum）的作用：对于那些当前的梯度方向与上一次梯度方向相同的参数，那么进行加强，即这些方向上更快了；对于那些当前的梯度方向与上一次梯度方向不同的参数，那么进行削减，即在这些方向上减慢了。

个人浅显理解，做个记录而已，不作参考，举例来说，假设设置的 learning rate 是随着迭代不断减小的，那么随着迭代，学习率是会减小的，但是设置了 momentum 后，在当前的梯度方向与上一次梯度方向相同的情况下，这个`权重（或是参数）更新的速度`更快了。