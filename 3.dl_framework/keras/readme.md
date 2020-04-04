# keras 学习

- [【笔记】Keras 学习笔记.md](./keras-learning.md)  [荐]★★★
- [主页 - Keras 中文文档](<https://keras.io/zh/>)
- [详解keras的model.summary()输出参数Param计算过程](<https://blog.csdn.net/ybdesire/article/details/85217688>) - 关于 console 台打印的参数数量的计算。
- [tensorflow - What does 'non-trainable params' mean? - Stack Overflow](<https://stackoverflow.com/questions/47312219/what-does-non-trainable-params-mean>) - console 台打印的 non-trainable params 的理解。



## 模型可视化

- [模型可视化 - Keras中文文档](<https://keras-cn.readthedocs.io/en/latest/Other/visualization/>)  |  [可视化 Visualization - Keras 中文文档](<https://keras.io/zh/visualization/>)

## 迁移学习/预训练

- [预训练模型Application - Keras中文文档](<https://keras-cn-twkun.readthedocs.io/Other/application/>)

## 导入模型测试

1、[Keras加载预训练模型 - 豌豆ip代理](<https://www.wandouip.com/t5i44145/>)  [荐]

``` 
比如训练模型的时候用到了自定义的模块AttentionLayer，那么在加载模型的时候需要在custom_objects的参数中声明对应的字典项，否则将会报模块未定义的错误。

model = load_model('./model1/GRUAttention( 0.8574).h5', custom_objects={'AttentionLayer': AttentionLayer})
在训练的过程中有时候也会用到自定义的损失函数，这时候如果你加载模型知识为了进行预测不再其基础上再进行训练，那么加载模型的时候就没有必要在custom_objects参数中声明对应的字典项，只需要将compile参数设为False即可：

model = load_model('./model1/GRUAttention(0.8574).h5', compile=False})
如果此时你好需要在加载后的模型上继续进行训练，那么声明损失函数对应的字典项就是必须的：

model = load_model('./model1/GRUAttention(0.8574).h5', compile=True, custom_objects={'focal_loss_fixed':focal_loss})
```



## 使用多 GPU

- [如何让keras训练深度网络时使用两张显卡？ - 知乎](<https://www.zhihu.com/question/67239897>)
- [keras 关于使用多个 gpu](<https://blog.csdn.net/MachineRandy/article/details/80040765>)  |  [Keras同时用多张显卡训练网络 - 简书](<https://www.jianshu.com/p/db0ba022936f>)
- [Keras多GPU及分布式](<https://blog.csdn.net/qq_34564612/article/details/79209965>) - 有两种方法可以在多张 GPU 上运行一个模型：数据并行/设备并行。大多数情况下，你需要的很可能是“数据并行”。

## 相关文章

1、[Keras中的多分类损失函数categorical_crossentropy](<https://blog.csdn.net/u010412858/article/details/76842216>)

``` 
注意：当使用`categorical_crossentropy`损失函数时，你的标签应为多类模式，例如如果你有 10 个类别，每一个样本的标签应该是一个 10 维的向量，该向量在对应有值的索引位置为 1 其余为 0。

可以使用这个方法进行转换：

from keras.utils.np_utils import to_categorical
categorical_labels = to_categorical(int_labels, num_classes=None)
```

2、[keras中的keras.utils.to_categorical方法](<https://blog.csdn.net/nima1994/article/details/82468965>) - `to_categorical(y, num_classes=None, dtype='float32')`

```
将整型标签转为 onehot。y 为 int 数组，num_classes 为标签类别总数，大于 max(y)（标签从0开始的）。

返回：如果 num_classes=None，返回 len(y)*[max(y)+1]（维度，m*n表示m行n列矩阵，下同），否则为 len(y)*num_classes。说出来显得复杂，请看下面实例。
```

3、[keras中的回调函数](<https://blog.csdn.net/jiandanjinxin/article/details/77097910>)

4、[为何Keras中的CNN是有问题的，如何修复它们？ - 知乎](<https://zhuanlan.zhihu.com/p/73549089>) - 关于参数初始化的问题。

> 我们证明，初始化是模型中特别重要的一件事情，这一点你可能经常忽略。此外，文章还证明，即便像 Keras 这种卓越的库中的默认设置，也不能想当然拿来就用。