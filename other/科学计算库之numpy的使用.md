# numpy 学习

## 1. random 函数

### numpy.random.rand()

numpy.random.rand(d0,d1,…,dn)

- rand函数根据给定维度生成[0,1)之间的数据，包含0，不包含1
- dn表格每个维度
- 返回值为指定维度的array

np.random.rand(4,2)





### numpy.random.randn()



### numpy.random.randint()



### 生成[0,1)之间的浮点数



### numpy.random.choice()



### numpy.random.seed()

- np.random.seed() 的作用：使得随机数据可预测。
- 当我们设置相同的 seed，每次生成的随机数相同。如果不设置 seed，则每次会生成不同的随机数

``` python
np.random.seed(0)
np.random.rand(5)
```

``` xml
array([ 0.5488135 ,  0.71518937,  0.60276338,  0.54488318,  0.4236548 ])
```

``` python
np.random.seed(1676)
np.random.rand(5)
```

``` xml
array([ 0.39983389,  0.29426895,  0.89541728,  0.71807369,  0.3531823 ])
```

``` python
np.random.seed(1676)
np.random.rand(5)
```

``` xml
array([ 0.39983389,  0.29426895,  0.89541728,  0.71807369,  0.3531823 ])  
```

需要注意的是，seed 值的有效次数仅为一次，因此，若要保证每次产生的随机数相同，则需要在调用随机数函数之前再次使用相同的 seed 值。下面给出相应的案例，即：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190410203603.png)

在机器学习和深度学习中，如果要保证部分参数（比如W权重参数）的随机初始化值相同，可以采用这种方式来实现。——from：<https://blog.csdn.net/zenghaitao0128/article/details/78558233>

参考文章：

- [为什么你用不好Numpy的random函数？](https://www.cnblogs.com/lemonbit/p/6864179.html)

## 2. random和np.random区别和联系

在 python 中，有两个模块可以产生随机数： 

1. python 自带 random 包： 提供一些基本的随机数产生函数，可满足基本需要。
2. numpy.random：提供一些产生随机数的高级函数，满足高级需求。

1、random 介绍：

| 函数                    | 功能                                                         | 返回                             | 备注                |
| ----------------------- | ------------------------------------------------------------ | -------------------------------- | ------------------- |
| **random.random()**     | 生成一个 [0,1) 之间的均匀分布浮点数                          | 一个浮点数                       |                     |
| **random.uniform(a,b)** | 生成一个 [a,b] 之间的均匀分布的浮点数，相当于 a + (b-a) * random.random() | 一个浮点数                       | 可以存在 a>b 的情况 |
|                         | 。。。                                                       | 。。。                           | 。。。              |
| **random.shuffle(lst)** | 在原列表上将元素打乱洗牌，原始序列改变                       | 功能函数：返回原序列打乱后的list | lst: 只支持列表     |

2、numpy.random 介绍：

| 函数                    | 功能                                                         | 返回       | 备注                |
| ----------------------- | ------------------------------------------------------------ | ---------- | ------------------- |
| **random.random()**     | 生成一个 [0,1) 之间的均匀分布浮点数                          | 一个浮点数 |                     |
| **random.uniform(a,b)** | 生成一个 [a,b] 之间的均匀分布的浮点数，相当于 a + (b-a) * random.random() | 一个浮点数 | 可以存在 a>b 的情况 |
| 。。。                  | 。。。                                                       | 。。。     | 。。。              |

详细参考：[Python随机数小结——random和np.random的区别与联系](<https://blog.csdn.net/qq_20011607/article/details/82288561>)

## 3. unique 函数

a = np.unique(A)，对于一维数组或者列表，unique 函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表。

``` python

import numpy as np
A = [1, 2, 2, 5,3, 4, 3]
a = np.unique(A)
B= (1, 2, 2,5, 3, 4, 3)
b= np.unique(B)
C= ['fgfh','asd','fgfh','asdfds','wrh']
c= np.unique(C)
print(a)
print(b)
print(c)
#   输出为 [1 2 3 4 5]
# [1 2 3 4 5]
# ['asd' 'asdfds' 'fgfh' 'wrh']
```

参考：[Python中numpy库unique函数解析](<https://blog.csdn.net/yangyuwen_yang/article/details/79193770>)



## 4. numpy.expand_dims 的用法

其实感觉 expand_dims(a, axis) 就是在 axis 的那一个轴上把数据加上去，这个数据在 axis 这个轴的 0 位置。 

例如原本为一维的 2 个数据，axis=0，则 shape 变为(1,2)，axis=1 则 shape 变为 (2,1) ；再例如 原本为 (2,3)，axis=0，则 shape 变为(1,2,3)，axis=1 则 shape 变为(2,1,3)

参考：[5 python numpy.expand_dims的用法](<https://blog.csdn.net/qq_16949707/article/details/53418912>)

## 5. numpy.argmax()

numpy.argmax(a, axis=None, out=None)：返回沿轴 axis 最大值的索引。

Parameters: 

- a : array_like 数组 
  
- axis : int, 可选 ，默认情况下，索引的是平铺的数组，否则沿指定的轴。 
  
- out : array, 可选 ，如果提供，结果以合适的形状和类型被插入到此数组中。 

Returns: index_array : ndarray of ints 

索引数组。它具有与 a.shape 相同的形状，其中 axis 被移除。 

例子：

``` python
>>> a = np.arange(6).reshape(2,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.argmax(a)
5
>>> np.argmax(a, axis=0)#0代表列
array([1, 1, 1])
>>> np.argmax(a, axis=1)#1代表行
array([2, 2])
>>>
>>> b = np.arange(6)
>>> b[1] = 5
>>> b
array([0, 5, 2, 3, 4, 5])
>>> np.argmax(b) # 只返回第一次出现的最大值的索引
1
```

再来看：

``` python
import numpy as np
a = np.arange(6).reshape(2, 3)
a[1, 1] = 7
print(a)
print(np.argmax(a))
```

运行结果：

``` xml
[[0 1 2]
 [3 7 5]]
4
```

可以看到 np.argmax(a) 是平铺后最大值的位置。

## numpy.squeeze函数

**语法**：numpy.squeeze(a,axis = None)

 1）a表示输入的数组；
 2）axis用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错；
 3）axis的取值可为None 或 int 或 tuple of ints, 可选。若axis为空，则删除所有单维度的条目；
 4）返回值：数组
 5) 不会修改原数组；

作用：从数组的形状中删除单维度条目，即把shape中为1的维度去掉

引用：https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html

场景：在机器学习和深度学习中，通常算法的结果是可以表示向量的数组（即包含两对或以上的方括号形式[[]]），如果直接利用这个数组进行画图可能显示界面为空（见后面的示例）。我们可以利用squeeze（）函数将表示向量的数组转换为秩为1的数组，这样利用matplotlib库函数画图时，就可以正常的显示结果了。

例1：

``` python
#例1
import numpy as np

a  = np.arange(10).reshape(1,10)
a

# array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# a.shape
# (1, 10)

b = np.squeeze(a)
b

# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

b.shape

# (10,)
```

例2：

``` python
#例2
c  = np.arange(10).reshape(2,5)
c

# array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])

np.squeeze(c)

# array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])

```

例3：

``` python
#例3
d  = np.arange(10).reshape(1,2,5)
d

# array([[[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]]])

d.shape

# (1, 2, 5)

np.squeeze(d)

# array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])

np.squeeze(d).shape

# (2, 5)
```

**结论**：根据上述例1~3可知，np.squeeze（）函数可以删除数组形状中的单维度条目，即把shape中为1的维度去掉，但是对非单维的维度不起作用。

（剩下略。。。

详细参考：[Numpy库学习—squeeze()函数](<https://blog.csdn.net/zenghaitao0128/article/details/78512715>)



## 6. Python的range和numpy的arange函数

Python 中的 range，以及 numpy 包中的 arange 函数。

**range()函数：**

- 函数说明： range(start, stop[, step]) -> range object，根据start与stop指定的范围以及step设定的步长，生成一个序列。

  ``` python
  参数含义：start:计数从start开始。默认是从0开始。例如range（5）等价于range（0， 5）;
                end:技术到end结束，但不包括end.例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
                scan：每次跳跃的间距，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)
  函数返回的是一个range object
  ```

- 例子：

  ``` python
  >>> range(0,5) 			 	#生成一个range object,而不是[0,1,2,3,4] 
  range(0, 5)   
  >>> c = [i for i in range(0,5)] 	 #从0 开始到4，不包括5，默认的间隔为1
  >>> c
  [0, 1, 2, 3, 4]
  >>> c = [i for i in range(0,5,2)] 	 #间隔设为2
  >>> c
  [0, 2, 4]
  ```

- 若需要生成[ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9]

  ``` python
  >>> range(0,1,0.1)    #range中的setp 不能使float
  Traceback (most recent call last):
    File "<pyshell#5>", line 1, in <module>
      range(0,1,0.1)
  TypeError: 'float' object cannot be interpreted as an integer
  ```

**arrange()函数：**

- 函数说明

  ``` xml
  函数说明：arange([start,] stop[, step,], dtype=None)根据start与stop指定的范围以及step设定的步长，生成一个 ndarray。 dtype : dtype
          The type of the output array.  If `dtype` is not given, infer the data
          type from the other input arguments.
  ```

- 例子：

  ```python
    >>> np.arange(3)
      array([0, 1, 2])
      >>> np.arange(3.0)
      array([ 0.,  1.,  2.])
      >>> np.arange(3,7)
      array([3, 4, 5, 6])
      >>> np.arange (3,7,2)
      array([3, 5])
  ```

  ``` python
  >>> arange(0,1,0.1)
  array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
  ```

——from：[Python 中的range(),arange()函数](<https://blog.csdn.net/qianwenhong/article/details/41414809>)



## 7. numpy中的数据类型转换：dtype、astype的区别

**1、随便玩玩**

生成一个浮点数组

``` python
>>> a = np.random.random(4)

查看信息：
>>> a
array([ 0.0945377 ,  0.52199916,  0.62490646,  0.21260126])
>>> a.dtype
dtype('float64')
>>> a.shape
(4,)
```

改变 dtype，发现数组长度翻倍！

``` python
>>> a.dtype = 'float32'
>>> a
array([  3.65532693e+20,   1.43907535e+00,  -3.31994873e-25,
         1.75549972e+00,  -2.75686653e+14,   1.78122652e+00,
        -1.03207532e-19,   1.58760118e+00], dtype=float32)
>>> a.shape
(8,)
```

改变dtype，数组长度再次翻倍！

``` python
>>> a.dtype = 'float16'
>>> a
array([ -9.58442688e-05,   7.19000000e+02,   2.38159180e-01,
         1.92968750e+00,              nan,  -1.66034698e-03,
        -2.63427734e-01,   1.96875000e+00,  -1.07519531e+00,
        -1.19625000e+02,              nan,   1.97167969e+00,
        -1.60156250e-01,  -7.76290894e-03,   4.07226562e-01,
         1.94824219e+00], dtype=float16)
>>> a.shape
(16,)
```

改变dtype='float'，发现默认就是float64，长度也变回最初的4

``` python
>>> a.dtype = 'float'
>>> a
array([ 0.0945377 ,  0.52199916,  0.62490646,  0.21260126])
>>> a.shape
(4,)
>>> a.dtype
dtype('float64')
```

把a变为整数，观察其信息

``` python
>>> a.dtype = 'int64'
>>> a
array([4591476579734816328, 4602876970018897584, 4603803876586077261,
       4596827787908854048], dtype=int64)
>>> a.shape
(4,)
```

改变dtype，发现数组长度翻倍！

``` python
>>> a.dtype = 'int32'
>>> a
array([ 1637779016,  1069036447, -1764917584,  1071690807,  -679822259,
        1071906619, -1611419360,  1070282372])
>>> a.shape
(8,)
```

（剩下略。。。）

**2、换一种玩法**

很多时候我们用 numpy 从文本文件读取数据作为 numpy 的数组，默认的 dtype 是 float64。但是有些场合我们希望有些数据列作为整数。如果直接改 dtype='int' 的话，就会出错！原因如上，数组长度翻倍了！！！

下面的场景假设我们得到了导入的数据。我们的本意是希望它们是整数，但实际上是却是浮点数(float64)

``` python
>>> b = np.array([1., 2., 3., 4.])
>>> b.dtype
dtype('float64')
```

**用 astype(int) 得到整数，并且不改变数组长度**

``` python
>>> c = b.astype(int)
>>> c
array([1, 2, 3, 4])
>>> c.shape
(8,)
>>> c.dtype
dtype('int32')
```

如果直接改变 b 的 dtype 的话，b 的长度翻倍了，这不是我们想要的（当然如果你想的话）

``` python
>>> b
array([ 1.,  2.,  3.,  4.])

>>> b.dtype = 'int'
>>> b.dtype
dtype('int32')
>>> b
array([         0, 1072693248,          0, 1073741824,          0,
       1074266112,          0, 1074790400])
>>> b.shape
(8,)
```

3、结论

numpy 中的数据类型转换，不能直接改原数据的 dtype，只能用函数 astype()。

——from：[numpy数据类型dtype转换](https://www.cnblogs.com/hhh5460/p/5129032.html)

## 8. numpy函数 all() 和any() 比较矩阵

数组元素的比对，我们可以直接使用“==”进行比较，比如：

``` python
a=np.array([1,2,3])
b=np.array([1,2,1])
a==b

结果：
array([ True,True,False],dtype=bool)
```

但是当数组元素较多时，查看输出结果便变得很麻烦，这时我们可以使用 all() 方法，直接比对 a 矩阵和 b 矩阵的所有对应的元素是否相等。

而 any() 方法是查看两矩阵是否有一个对应元素相等。事实上，all() 操作就是对两个矩阵的比对结果再做一次与运算，而 any 则是做一次或运算。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190618111812.png)

