## 前言

本文主要为 Python基础入门笔记（一）内容的补充。



## 一、迭代器和生成器

### 1.1 Python迭代器

迭代器是一个可以记住遍历的位置的对象。

迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。

迭代器只能往前不会后退。

迭代器有两个基本的方法：`iter()` 和 `next()`，且字符串、列表或元组对象都可用于创建迭代器，迭代器对象可以使用常规 for 语句进行遍历，也可以使用 next() 函数来遍历。

具体的实例：
``` python 
# 1、字符创创建迭代器对象
str1 = 'jaybo'
iter1 = iter ( str1 )

# 2、list对象创建迭代器
list1 = [1,2,3,4]
iter2 = iter ( list1 )

# 3、tuple(元祖) 对象创建迭代器
tuple1 = ( 1,2,3,4 )
iter3 = iter ( tuple1 )

# for 循环遍历迭代器对象
for x in iter1 :
    print ( x , end = ' ' )

print('\n------------------------')

# next() 函数遍历迭代器
while True :
    try :
        print ( next ( iter3 ) )
    except StopIteration :
        break
```

最后输出的结果：
``` python
j a y b o
------------------------
1
2
3
4
```


**list(列表)生成式：** 

语法为：
``` python 
[expr for iter_var in iterable] 
[expr for iter_var in iterable if cond_expr]
```

第一种语法：首先迭代 iterable 里所有内容，每一次迭代，都把 iterable 里相应内容放到iter_var 中，再在表达式中应用该 iter_var 的内容，最后用表达式的计算值生成一个列表。

第二种语法：加入了判断语句，只有满足条件的内容才把 iterable 里相应内容放到 iter_var 中，再在表达式中应用该 iter_var 的内容，最后用表达式的计算值生成一个列表。

实例，用一句代码打印九九乘法表：
``` python 
print('\n'.join([' '.join ('%dx%d=%2d' % (x,y,x*y)  for x in range(1,y+1)) for y in range(1,10)]))
```
输出结果：
``` python
1x1= 1
1x2= 2 2x2= 4
1x3= 3 2x3= 6 3x3= 9
1x4= 4 2x4= 8 3x4=12 4x4=16
1x5= 5 2x5=10 3x5=15 4x5=20 5x5=25
1x6= 6 2x6=12 3x6=18 4x6=24 5x6=30 6x6=36
1x7= 7 2x7=14 3x7=21 4x7=28 5x7=35 6x7=42 7x7=49
1x8= 8 2x8=16 3x8=24 4x8=32 5x8=40 6x8=48 7x8=56 8x8=64
1x9= 9 2x9=18 3x9=27 4x9=36 5x9=45 6x9=54 7x9=63 8x9=72 9x9=81
```

### 1.2 生成器

在 Python 中，使用了 yield 的函数被称为生成器（generator）。

跟普通函数不同的是，**生成器是一个返回迭代器的函数，只能用于迭代操作，更简单点理解生成器就是一个迭代器。** 

在调用生成器运行的过程中，每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值。并在下一次执行 next() 方法时从当前位置继续运行。

**①创建：** 

生成器的创建：最简单最简单的方法就是把一个列表生成式的 [] 改成 ()
``` python
gen= (x * x for x in range(10))
print(gen)
```
输出结果：
``` python
generator object  at 0x0000000002734A40
```

创建 List 和 generator 的区别仅在于最外层的 [] 和 () 。但是生成器并不真正创建数字列表， 而是返回一个生成器，这个生成器在每次计算出一个条目后，把这个条目“产生” ( yield ) 出来。 生成器表达式使用了“惰性计算” ( lazy evaluation，也有翻译为“延迟求值”，我以为这种按需调用 call by need 的方式翻译为惰性更好一些)，只有在检索时才被赋值（ evaluated ），所以在列表比较长的情况下使用内存上更有效。


**②以函数形式实现生成器：** 

其实生成器也是一种迭代器，但是你只能对其迭代一次。这是因为它们并没有把所有的值存在内存中，而是在运行时生成值。你通过遍历来使用它们，要么用一个“for”循环，要么将它们传递给任意可以进行迭代的函数和结构。而且实际运用中，大多数的生成器都是通过函数来实现的。

生成器和函数的不同：
> 函数是顺序执行，遇到 return 语句或者最后一行函数语句就返回。而变成 generator 的函数，在每次调用 next() 的时候执行，遇到 yield语句返回，再次执行时从上次返回的 yield 语句处继续执行。

举个例子：
``` python
def odd():
    print ( 'step 1' )
    yield ( 1 )
    print ( 'step 2' )
    yield ( 3 )
    print ( 'step 3' )
    yield ( 5 )

o = odd()
print( next( o ) ) 
print( next( o ) ) 
print( next( o ) )
```
输出结果：
``` python
step 1
1
step 2
3
step 3
5
```

可以看到，odd 不是普通函数，而是 generator，在执行过程中，遇到 yield 就中断，下次又继续执行。执行 3 次 yield 后，已经没有 yield 可以执行了，如果你继续打印 print( next( o ) ) ,就会报错的。所以通常在 generator 函数中都要对错误进行捕获。

打印杨辉三角：
``` python 
def triangles( n ):         # 杨辉三角形
    L = [1]
    while True:
        yield L
        L.append(0)
        L = [ L [ i -1 ] + L [ i ] for i in range (len(L))]

n= 0
for t in triangles( 10 ):   # 直接修改函数名即可运行
    print(t)
    n = n + 1
    if n == 10:
        break
```
输出结果：
``` python 
[1]
[1, 1]
[1, 2, 1]
[1, 3, 3, 1]
[1, 4, 6, 4, 1]
[1, 5, 10, 10, 5, 1]
[1, 6, 15, 20, 15, 6, 1]
[1, 7, 21, 35, 35, 21, 7, 1]
[1, 8, 28, 56, 70, 56, 28, 8, 1]
[1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
```

### 1.3 延伸

①反向迭代

使用 Python 中有内置的函数 `reversed()`。

要注意一点就是：**反向迭代仅仅当对象的大小可预先确定或者对象实现了 `__reversed__()` 的特殊方法时才能生效。 如果两者都不符合，那你必须先将对象转换为一个列表才行。** 


②同时迭代多个序列

为了同时迭代多个序列，使用 zip() 函数，具体示例：
``` python
names = ['jaychou', 'zjl', '周杰伦']
ages = [18, 19, 20]
for name, age in zip(names, ages):
     print(name,age)
```
输出的结果：
``` python
jaychou 18
zjl 19
周杰伦 20
```

其实 `zip(a, b)` 会生成一个可返回元组 (x, y) 的迭代器，其中 x 来自 a，y 来自 b。 一旦其中某个序列到底结尾，迭代宣告结束。 因此迭代长度跟参数中最短序列长度一致。注意理解这句话，**也就是说如果 a ， b 的长度不一致的话，以最短的为标准，遍历完后就结束。** 



## 二、模块与包

### 2.1 模块

#### 2.1.1 什么是模块

在 Python 中，一个 `.py` 文件就称之为一个模块（Module）。

我们学习过函数，知道函数是实现一项或多项功能的一段程序 。其实模块就是函数功能的扩展。为什么这么说呢？那是因为模块其实就是实现一项或多项功能的程序块。

通过上面的定义，不难发现，**函数和模块都是用来实现功能的，只是模块的范围比函数广，在模块中，可以有多个函数。**

模块的好处：

- 模块使用的最大好处是大大提高了代码的可维护性，当然，还提高了代码的复用性。

- 使用模块还可以避免函数名和变量名冲突，相同名字的变量完全可以分别存在不同的模块中。

  > PS：但是也要注意，变量的名字尽量不要与内置函数名字冲突。常见的内置函数：[链接直达](https://docs.python.org/3/library/functions.html)

再这也顺带先延伸下关于包的内容吧：

> 当编写的模块多了，模块的名字重复的概率就增加了。如何解决这个问题呢？
>
> Python 引入了按目录来组织模块，称为包（Package），比如：
>
> ``` xml
> extensions
> ├─ __init__.py
> ├─ dog.py
> └─ cat.py
> ```
>
> 现在 `dog.py` 模块的名字就变成了 `extensions.dog`。
>
> PS：请注意，每一个 package 目录下面都会有一个`__init__.py` 的文件，这个文件是必须有的，否则， Python 就把这个目录当成普通目录，而不是一个 package directory。
>
> 另外如何使用包中的模块（Module）呢？如下编写一个`dog.py`模块：
>
> ``` python
> #!/usr/bin/env python3
> # -*- coding: utf-8 -*-
>
> ' a test module '
>
> __author__ = 'jack guo'
>
> import sys
>
> def shout():
>     args = sys.argv
>     if len(args)==1:
>         print('Hello, I'm afei, welcome to world!')
>     elif len(args)==2:
>         print('Hello, %s!' % args[1])
>    else:
>         print('Yes,sir')
>
> if __name__=='__main__':
>     shout()
> ```
>
> 解释下：
>
> ``` xml
> 第1行注释可以让dog.py文件直接在linux上运行；
> 第2行注释表示.py文件本身使用标准UTF-8编码；
> 第4行表示模块的文档注释；
> 第6行表示模块的作者；
>
> 注意最后两行代码，当我们调试dog.py时，shout()会调用，当在其他模块导入dog.py时，shout()不执行。
> ```
>
> 模块的一种标准模板：
>
> ``` python
> #!/usr/bin/env python3
> # -*- coding: utf-8 -*-
>
> ' a test module '
>
> __author__ = 'jack guo'
> ```
>
> 以上是模块的标准模板，当然，你也可以不这样做。

#### 2.1.2 模块的导入

导入模块我们使用关键字 `import`，语法格式如下：`import module1[, module2[,... moduleN]`

如：`import math` 导入标准模块中的 math 模块。

一个模块只会被导入一次，不管你执行了多少次 `import`。这样可以防止导入模块被一遍又一遍地执行。

Python 解释器是怎样找到对应的文件的呢？

搜索路径：由一系列目录名组成的。Python 解释器就依次从这些目录中去寻找所引入的模块。这看起来很像环境变量，事实上，也可以通过定义环境变量的方式来确定搜索路径。搜索路径是在 Python 编译或安装的时候确定的，安装新的库应该也会修改。搜索路径被存储在 sys 模块中的 path 变量 。可以打印出来：
``` python 
import sys

print(sys.path)
```

#### 2.1.3 导入模块中的属性和方法及调用

①导入模块的方法 

- `import 模块名`
- `import 模块名 as 新名字`
- `from 模块名 import 函数名`：大型项目中应尽量避免使用此方法，除非你非常确定不会造成命名冲突；它有一个好处就是可直接使用`function()`而不用加`module.function()`了。

> PS1：导入模块并不意味着在导入时执行某些操作，它们主要用于定义，比如变量、函数和类等。
>
> PS2：可以使用 `from ··· import *` 语句把某个模块中的所有方法属性都导入。

②模块中变量、函数以及类的属性和方法的调用

1. `module.variable`
2. `module.function()`
3. `module.class.variable`

#### 2.1.4 模块的搜索路径sys模块的使用）

（1）程序所在目录

（2）标准库的安装路径

（3）操作系统环境变量 PYTHONPATH 指向的路径

- 获得当前 Python 搜索路径的方法：

  ``` python
  import sys
  print(sys.path)
  ```

  输出：

  ``` xml
  ['D:\\workspace_pycharm', 'D:\\workspace_pycharm', 'D:\\python-practice', 'D:\\devInstall\\devPython\\Python36\\python36.zip', 'D:\\devInstall\\devPython\\Python36\\DLLs', 'D:\\devInstall\\devPython\\Python36\\lib', 'D:\\devInstall\\devPython\\Python36', 'D:\\devInstall\\devPython\\Python36\\lib\\site-packages']
  ```

- sys 模块的 argv 变量的用法： 

  - sys 模块有一个 `argv`(argument values) 变量，用 list 存储了命令行的所有参数。
  - `argv` 至少有一个元素，因为第一个元素永远都是`.py`文件的名称。

  ``` python
  $ python solve.py 0    # 命令行语句
  # 获得argv变量的值
  sys.argv = ['solve.py', '0']
  sys.argv[0] = 'solve.py'
  sys.argv[1] = '0'
  ```

#### 2.1.5 主模块和非主模块

在 Python 函数中，如果一个函数调用了其他函数完成一项功能，我们称这个函数为主函数，如果一个函数没有调用其他函数，我们称这种函数为非主函数。主模块和非主模块的定义也类似，如果一个模块被直接使用，而没有被别人调用，我们称这个模块为主模块，如果一个模块被别人调用，我们称这个模块为非主模块。

怎么区分主模块和非主模块呢？

可以利用 `__name__`属性。如果一个属性的值是 `__main__` ，那么就说明这个模块是主模块，反之亦然。但是要注意了：这个 `__main__` 属性只是帮助我们判断是否是主模块，并不是说这个属性决定他们是否是主模块，决定是否是主模块的条件只是这个模块有没有被人调用。如下：

``` python 
if __name__ == '__main__':
    print('main')
else:
    print('not main')
```
如果输出结果为 main 则该模块为主模块。

**!!!补充：** 在初学 Python 过程中，总能遇到  `if __name__ == 'main' `语句，我们正好来好好了解下。

先举例子，假如 A.py 文件内容如下：

``` python
def sayhello():
    print('Hello!')
print('Hi!')
print(__name__)
```

输出结果：

``` xml
Hi!
__main__
```

结果很简单，说明在运行 A.py 本身文件时，变量`__name__`的值是`__main__`。

现有个 B.py 文件，代码如下：

``` python
import A
A.sayhello()
print('End')
```

可以看到，在 B.py 文件中，模块 A 被导入，运行结果如下：

``` xml
Hi!
A
Hello!
End
```

这里涉及一些语句运行顺序问题，在 B.py 文件中，模块 A 中的 sayhello 函数是调用时才执行的，但是 A 中的 print 语句会立刻执行（因为没有缩进，因此与def是平行级别的）。因此会先依次执行：

``` python
print('Hi!')
print(__name__)
```

然后执行：

``` python
A.sayhello()
print('End')
```

运行结果中`Hi!`对应于 A 模块中的 `print('Hi!')`，而结果 A 对应于 `print(__name__)`，可见当在 B 文件中调用 A 模块时，变量`__name__`的值由`__main__`变为了模块 A 的名字。

这样的好处是我们可以在 A.py 文件中进行一些测试，而避免在模块调用的时候产生干扰，比如将 A 文件改为：

``` python
def sayhello():
    print('Hello!')
print('Hi!')
print(__name__)

if __name__ == '__main__':
    print('I am module A')
```

再次单独运行 A.py 文件时，结果中会多出`I am module A`语句：

``` xml
Hi!
__main__
I am module A
```

而运行 B.py 文件，即调用 A 模块时，却不会显示该语句：

``` xml
Hi!
A
Hello!
End
```

简短总结下：

> 模块属性`__name__`，它的值由 Python 解释器设定。如果 Python 程序是作为主程序调用，其值就设为`__main__`，如果是作为模块被其他文件导入，它的值就是其文件名。
>
> 每个模块都有自己的私有符号表，所有定义在模块里面的函数把它当做全局符号表使用。

### 2.2 包

#### 2.2.1 什么是包

我们自己在编写模块时，不必考虑名字会与其他模块冲突。但是也要注意，尽量不要与内置函数名字冲突。但是这里也有个问题，如果不同的人编写的模块名相同怎么办？为了避免模块名冲突，Python 又引入了按目录来组织模块的方法，称为包（Package）。

仔细观察的人，基本会发现，每一个包目录下面都会有一个 `__init__.py` 的文件。这个文件是必须的，否则，Python 就把这个目录当成普通目录，而不是一个包。 `__init__.py` 可以是空文件，也可以有 Python 代码，因为 `__init__.py` 本身就是一个模块，而它对应的模块名就是它的包名。

#### 2.2.2 包的定义和优点

- Python 把**同类的模块**放在一个文件夹中统一管理，这个文件夹称之为一个**包**。
- 如果把所有模块都放在一起显然不好管理，并且有命名冲突的可能。
- 包其实就是把模块**分门别类**地存放在不同的文件夹，然后把各个文件夹的位置告诉Python。
- Python 的包是按**目录**来组织模块的，也可以有**多级目录**，组成多级层次的包结构。

#### 2.2.3 包的创建

- 创建一个文件夹，用于存放相关的模块，**文件夹的名字即为包的名字**。
- 在文件夹中创建一个`__init__.py`的模块文件，内容可以为空(普通文件夹和包的区别)。
- 将相关模块放入文件夹中

#### 2.3.4 包的存放路径及包中模块的导入与调用

①包的存放

- 如果不想把相关的模块文件放在所创建的文件夹中，那么最好的选择就是：放在默认的`site-packages`文件夹里，因为它就是用来存放你的模块文件的。
- `sys.path.append(‘模块的存放位置’)`只是在运行时生效，运行结束后失效。
- 将包的存放路径加入用户系统环境变量中的 PYTHONPYTH 中去，这样在任何位置都可以调用包了（推荐）。

②包中模块的导入

1. `import 包名.模块名`
2. `import 包名.模块名 as 新名字`
3. `from 包名 import 模块名`

③包中模块的变量、函数以及类的属性和方法的调用

1. `package.module.variable`
2. `package.module.function()`
3. `package.module.class.variable`

### 2.3 作用域

学习过 Java 的同学都知道，Java 的类里面可以给方法和属性定义公共的（ public ）或者是私有的 （ private ）,这样做主要是为了我们希望有些函数和属性能给别人使用或者只能内部使用。 通过学习 Python 中的模块，其实和 Java 中的类相似，那么我们怎么实现在一个模块中，有的函数和变量给别人使用，有的函数和变量仅仅在模块内部使用呢？

在 Python 中，是通过 `_` 前缀来实现的。正常的函数和变量名是公开的（public），可以被直接引用，比如：`abc`，`ni12`，`PI` 等。

类似`__xxx__`这样的变量是特殊变量，可以被直接引用，但是有特殊用途，比如上面的 `__name__` 就是特殊变量，还有 `__author__` 也是特殊变量，用来标明作者。注意，我们自己的变量一般不要用这种变量名；**类似` _xxx` 和 `__xxx` 这样的函数或变量就是非公开的（private），不应该被直接引用，比如 `_abc` ，`__abc` 等.** 

注意：这里是说不应该，而不是不能。因为 Python 种并没有一种方法可以完全限制访问 private 函数或变量，但是，从编程习惯上不应该引用 private 函数或变量。



## 三、面向对象

Python 对属性的访问控制是靠程序员自觉的。

我们也可以把方法看成是类的属性的，那么方法的访问控制也是跟属性是一样的，也是没有实质上的私有方法。一切都是靠程序员自觉遵守 Python 的编程规范。

### 3.1 类

#### 3.1.1 方法的装饰器

- `@classmethod`：调用的时候直接使用类名类调用，而不是某个对象

- `@property`：可以像访问属性一样调用方法

``` python 
class UserInfo:

    ...

    @classmethod
    def get_name(cls):
        return cls.lv

    @property
    def get_age(self):
        return self._age

    if __name__ == '__main__':   
        ...

        # 直接使用类名类调用，而不是某个对象
        print(UserInfo.lv)
        # 像访问属性一样调用方法（注意看get_age是没有括号的）
        print(userInfo.get_age)
```

#### 3.1.2 继承

语法格式：
``` python 
class ClassName(BaseClassName):
    <statement-1>
    .
    .
    .
    <statement-N>
```

当然上面的是单继承，**Python 也是支持多继承的**（注意： Java 是单继承、多实现），具体的语法如下：
``` python 
class ClassName(Base1,Base2,Base3):
    <statement-1>
    .
    .
    .
    <statement-N>
```

**多继承有一点需要注意的：若是父类中有相同的方法名，而在子类使用时未指定，Python 在圆括号中父类的顺序，从左至右搜索 ， 即方法在子类中未找到时，从左到右查找父类中是否包含方法。**

继承的子类的好处：

- 会继承父类的属性和方法
- 可以自己定义，覆盖父类的属性和方法


#### 3.1.3 多态

看个例子就好了：

``` python
class User(object):
    def __init__(self, name):
        self.name = name

    def printUser(self):
        print('Hello !' + self.name)

class UserVip(User):
    def printUser(self):
        print('Hello ! 尊敬的Vip用户：' + self.name)

class UserGeneral(User):
    def printUser(self):
        print('Hello ! 尊敬的用户：' + self.name)

def printUserInfo(user):
    user.printUser()

if __name__ == '__main__':
    userVip = UserVip('大金主')
    printUserInfo(userVip)
    userGeneral = UserGeneral('水货')
    printUserInfo(userGeneral)
```
输出结果：
``` python 
Hello ! 尊敬的Vip用户：大金主
Hello ! 尊敬的用户：水货
```

可以看到，userVip 和 userGeneral 是两个不同的对象，对它们调用 printUserInfo 方法，它们会自动调用实际类型的 printUser 方法，作出不同的响应。这就是多态的魅力。

> PS：有了继承，才有了多态，也会有不同类的对象对同一消息会作出不同的相应。

#### 3.1.4 Python中的魔法方法

在 Python 中，所有以 "**" 双下划线包起来的方法，都统称为"魔术方法"。比如我们接触最多的 `init__` 。魔术方法有什么作用呢？

使用这些魔术方法，我们可以构造出优美的代码，将复杂的逻辑封装成简单的方法。

我们可以使用 Python 内置的方法 `dir()` 来列出类中所有的魔术方法。示例如下：

``` python
class User(object):
    pass


if __name__ == '__main__':
    print(dir(User()))
```

输出的结果：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-9-15-57508065.jpg)

可以看到，一个类的魔术方法还是挺多的，截图没有截全。不过我们只需要了解一些常见和常用的魔术方法就好了。

**1、属性的访问控制** 

Python 没有真正意义上的私有属性。然后这就导致了对 Python 类的封装性比较差。我们有时候会希望 Python 能够定义私有属性，然后提供公共可访问的 get 方法和 set 方法。Python 其实可以通过魔术方法来实现封装。

| 方法                             | 说明                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| `__getattr__(self, name)`        | 该方法定义了你试图访问一个不存在的属性时的行为。因此，重载该方法可以实现捕获错误拼写然后进行重定向，或者对一些废弃的属性进行警告。 |
| `__setattr__(self, name, value)` | 定义了对属性进行赋值和修改操作时的行为。不管对象的某个属性是否存在,都允许为该属性进行赋值。有一点需要注意，实现 `__setattr__` 时要避免"无限递归"的错误 |
| `__delattr__(self, name)`        | `__delattr__` 与 `__setattr__` 很像，只是它定义的是你删除属性时的行为。实现 `__delattr__` 是同时要避免"无限递归"的错误 |
| `__getattribute__(self, name)`   | `__getattribute__` 定义了你的属性被访问时的行为，相比较，`__getattr__` 只有该属性不存在时才会起作用。因此，在支持 `__getattribute__`的 Python 版本，调用`__getattr__` 前必定会调用 `__getattribute__`，`__getattribute__` 同样要避免"无限递归"的错误。 |

**2、对象的描述器** 

一般来说，一个描述器是一个有“绑定行为”的对象属性 (object attribute)，它的访问控制被描述器协议方法重写。这些方法是 `__get__()`，`__set__() `和 `__delete__()`。有这些方法的对象叫做描述器。

默认对属性的访问控制是从对象的字典里面 （`__dict__）` 中获取 （get） , 设置 （set） 和删除 （delete） 。举例来说， `a.x` 的查找顺序是 `a.__dict__['x']`，然后 `type(a).__dict__['x']`，然后找 `type(a)` 的父类 ( 不包括元类 (metaclass) ）。如果查找到的值是一个描述器，Python 就会调用描述器的方法来重写默认的控制行为。这个重写发生在这个查找环节的哪里取决于定义了哪个描述器方法。注意，只有在新式类中时描述器才会起作用。

至于新式类最大的特点就是所有类都继承自 type 或者 object 的类。

在面向对象编程时，如果一个类的属性有相互依赖的关系时，使用描述器来编写代码可以很巧妙的组织逻辑。在 Django 的 ORM 中，`models.Model` 中的 InterField 等字段，就是通过描述器来实现功能的。

看一个例子：
``` python
class User(object):
    def __init__(self, name='小明', sex='男'):
        self.sex = sex
        self.name = name

    def __get__(self, obj, objtype):
        print('获取 name 值')
        return self.name

    def __set__(self, obj, val):
        print('设置 name 值')
        self.name = val

class MyClass(object):
    x = User('小明', '男')
    y = 5

if __name__ == '__main__':
    m = MyClass()
    print(m.x)

    print('\n')

    m.x = '大明'
    print(m.x)

    print('\n')

    print(m.x)

    print('\n')

    print(m.y)
```
输出结果：
``` xml
获取 name 值
小明


设置 name 值
获取 name 值
大明


获取 name 值
大明


5
```

3、自定义容器（Container）

我们知道在 Python 中，常见的容器类型有：dict、tuple、list、string。其中也提到过可容器和不可变容器的概念。其中 tuple、string 是不可变容器，dict、list 是可变容器。

可变容器和不可变容器的区别在于，不可变容器一旦赋值后，不可对其中的某个元素进行修改。

那么这里先提出一个问题，这些数据结构就够我们开发使用吗？不够的时候，或者说有些特殊的需求不能单单只使用这些基本的容器解决的时候，该怎么办呢？

这个时候就需要自定义容器了，那么具体我们该怎么做呢？

| 功能                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 自定义不可变容器类型                                         | 需要定义 `__len__` 和 `__getitem__`方法                      |
| 自定义可变类型容器                                           | 在不可变容器类型的基础上增加定义 `__setitem__` 和 `__delitem__` |
| 自定义的数据类型需要迭代                                     | 需定义 `__iter__`                                            |
| 返回自定义容器的长度                                         | 需实现 `__len__(self)`                                       |
| 自定义容器可以调用 `self[key]`，如果 key 类型错误，抛出 TypeError，如果没法返回 key对应的数值时,该方法应该抛出 ValueError | 需要实现 `__getitem__(self, key)`                            |
| 当执行 `self[key] = value` 时                                | 调用是 `__setitem__(self, key, value)`这个方法               |
| 当执行 `del self[key]` 方法                                  | 其实调用的方法是 `__delitem__(self, key)`                    |
| 当你想你的容器可以执行 `for x in container:` 或者使用 `iter(container)` 时 | 需要实现 `__iter__(self)` ，该方法返回的是一个迭代器         |

还有很多魔术方法，比如运算符相关的模式方法，就不在该文展开了。

### 3.2 枚举类

#### 3.2.1 什么是枚举

举例，直接看代码：
``` python 
from enum import Enum

Month = Enum('Month1', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))

# 遍历枚举类型
for name, member in Month.__members__.items():
    print(name, '---------', member, '----------', member.value)

# 直接引用一个常量
print('\n', Month.Jan)
```
输出结果：
``` python 
Jan --------- Month1.Jan ---------- 1
Feb --------- Month1.Feb ---------- 2
Mar --------- Month1.Mar ---------- 3
Apr --------- Month1.Apr ---------- 4
May --------- Month1.May ---------- 5
Jun --------- Month1.Jun ---------- 6
Jul --------- Month1.Jul ---------- 7
Aug --------- Month1.Aug ---------- 8
Sep --------- Month1.Sep ---------- 9
Oct --------- Month1.Oct ---------- 10
Nov --------- Month1.Nov ---------- 11
Dec --------- Month1.Dec ---------- 12

Month.Jan
```

可见，我们可以直接使用 Enum 来定义一个枚举类。上面的代码，我们创建了一个有关月份的枚举类型 Month，这里要注意的是构造参数，第一个参数 Month 表示的是该枚举类的类名，第二个 tuple 参数，表示的是枚举类的值； 当然，枚举类通过 `__members__` 遍历它的所有成员的方法。

注意的一点是 ， member.value 是自动赋给成员的 int 类型的常量，默认是从 1 开始的。而且 Enum 的成员均为单例（Singleton），并且不可实例化，不可更改。

#### 3.2.2 自定义枚举类型

有时候我们需要控制枚举的类型，那么我们可以 Enum 派生出自定义类来满足这种需要。修改上面的例子：
``` python 
from enum import Enum, unique

Enum('Month1', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))

# @unique 装饰器可以帮助我们检查保证没有重复值
@unique
class Month1(Enum):
    Jan = 'January'
    Feb = 'February'
    Mar = 'March'
    Apr = 'April'
    May = 'May'
    Jun = 'June'
    Jul = 'July'
    Aug = 'August'
    Sep = 'September '
    Oct = 'October'
    Nov = 'November'
    Dec = 'December'

if __name__ == '__main__':
    print(Month1.Jan, '----------',
          Month1.Jan.name, '----------', Month1.Jan.value)
    for name, member in Month1.__members__.items():
        print(name, '----------', member, '----------', member.value)
```
输出结果：
``` python 
Month1.Jan ---------- Jan ---------- January
Jan ---------- Month1.Jan ---------- January
Feb ---------- Month1.Feb ---------- February
Mar ---------- Month1.Mar ---------- March
Apr ---------- Month1.Apr ---------- April
May ---------- Month1.May ---------- May
Jun ---------- Month1.Jun ---------- June
Jul ---------- Month1.Jul ---------- July
Aug ---------- Month1.Aug ---------- August
Sep ---------- Month1.Sep ---------- September 
Oct ---------- Month1.Oct ---------- October
Nov ---------- Month1.Nov ---------- November
Dec ---------- Month1.Dec ---------- December
```

#### 4.2.3 枚举类的比较

因为枚举成员不是有序的，所以它们只支持通过标识(identity) 和相等性 (equality) 进行比较。下面来看看 `==` 和 `is` 的使用：
``` python 
from enum import Enum

class User(Enum):
    Twowater = 98
    Liangdianshui = 30
    Tom = 12

Twowater = User.Twowater
Liangdianshui = User.Liangdianshui

print(Twowater == Liangdianshui, Twowater == User.Twowater)
print(Twowater is Liangdianshui, Twowater is User.Twowater)

try:
    print('\n'.join('  ' + s.name for s in sorted(User)))
except TypeError as err:
    print(' Error : {}'.format(err))
```
输出结果：
``` xml
False True
False True
 Error : '<' not supported between instances of 'User' and 'User'
```
可以看看最后的输出结果，报了个异常，那是因为大于和小于比较运算符引发 TypeError 异常。也就是 Enum 类的枚举是不支持大小运算符的比较的。

但是使用 IntEnum 类进行枚举，就支持比较功能。
``` python 
import enum

class User(enum.IntEnum):
    Twowater = 98
    Liangdianshui = 30
    Tom = 12
    
try:
    print('\n'.join(s.name for s in sorted(User)))
except TypeError as err:
    print(' Error : {}'.format(err))
```
输出结果：
``` xml
Tom
Liangdianshui
Twowater
```
通过输出的结果可以看到，枚举类的成员通过其值得大小进行了排序。也就是说可以进行大小的比较。

### 3.3 元类

#### 3.3.1 Python 中类也是对象

在大多数编程语言中，类就是一组用来描述如何生成一个对象的代码段。在 Python 中这一点也是一样的。但是，Python 中的类有一点跟大多数的编程语言不同，在 Python 中，**可以把类理解成也是一种对象。对的，这里没有写错，就是对象。** 

因为只要使用关键字 `class`，Python 解释器在执行的时候就会创建一个对象。如：

``` python 
class ObjectCreator(object):
    pass
```
当程序运行这段代码的时候，就会在内存中创建一个对象，名字就是ObjectCreator。这个对象（类）自身拥有创建对象（类实例）的能力，而这就是为什么它是一个类的原因。

#### 3.3.2 使用type()动态创建类

因为类也是对象，所以我们可以在程序运行的时候创建类。Python 是动态语言。动态语言和静态语言最大的不同，就是函数和类的定义，不是编译时定义的，而是运行时动态创建的。在之前，我们先了了解下 `type()` 函数。

``` python 
class Hello(object):
    def hello(self, name='Py'):
        print('Hello,', name)
```
然后再另外一个模块引用 hello 模块，输出相应信息。（其中 `type()` 函数的作用是可以查看一个类型和变量的类型。)
``` python 
from com.strivebo.hello import Hello

h = Hello()
h.hello()

print(type(Hello))
print(type(h))
```
输出信息：
``` xml
Hello, Py
<class 'type'>
<class 'com.twowater.hello.Hello'>
```
上面也提到过，`type()` 函数可以查看一个类型或变量的类型，Hello 是一个 class ，它的类型就是 type ，而 h 是一个实例，它的类型就是 `com.strivebo.hello.Hello`。前面的 `com.strivebo` 是我的包名，hello 模块在该包名下。

在这里还要细想一下，上面的例子中，我们使用 `type()` 函数查看一个类型或者变量的类型。其中查看了一个 Hello class 的类型，打印的结果是： `<class 'type'> `。其实 `type()` 函数不仅可以返回一个对象的类型，也可以创建出新的类型。class 的定义是运行时动态创建的，而创建 class 的方法就是使用 `type()` 函数。比如我们可以通过 `type()` 函数创建出上面例子中的 Hello 类，具体看下面的代码：

``` python 
def printHello(self, name='Py'):
    # 定义一个打印 Hello 的函数
    print('Hello,', name)

# 创建一个 Hello 类
Hello = type('Hello', (object,), dict(hello=printHello))

# 实例化 Hello 类
h = Hello()
# 调用 Hello 类的方法
h.hello()
# 查看 Hello class 的类型
print(type(Hello))
# 查看实例 h 的类型
print(type(h))
```
输出结果：
``` xml
Hello, Py
<class 'type'>
<class '__main__.Hello'>
```
在这里，需先了解下通过 `type()` 函数创建 class 对象的参数说明：

1. class 的名称，比如例子中的起名为 `Hello`
2. 继承的父类集合，注意 Python 支持多重继承，如果只有一个父类，tuple 要使用单元素写法；例子中继承 object 类，因为是单元素的 tuple ，所以写成 `(object,)`
3. class 的方法名称与函数绑定；例子中将函数 `printHello` 绑定在方法名 `hello` 中

具体的模式如下：`type(类名, 父类的元组(针对继承的情况，可以为空)，包含属性的字典(名称和值))` 

好了，了解完具体的参数使用之外，我们看看输出的结果，可以看到，通过 `type()` 函数创建的类和直接写 class 是完全一样的，因为 Python 解释器遇到 class 定义时，仅仅是扫描一下 class 定义的语法，然后调用 `type()` 函数创建出 class 的。

#### 3.3.3 什么是元类

我们创建类的时候，大多数是为了创建类的实例对象。那么元类呢？元类就是用来创建类的。也可以换个理解方式就是：**元类就是类的类。** 

通过上面 `type()` 函数的介绍，我们知道可以通过 `type()` 函数创建类：`MyClass = type('MyClass', (), {})`

**实际上 type() 函数是一个元类。`type()` 就是 Python 在背后用来创建所有类的元类。**

那么现在我们也可以猜到一下为什么 `type()` 函数是 type 而不是 Type呢？

> 这可能是为了和 str 保持一致性，str 是用来创建字符串对象的类，而 int 是用来创建整数对象的类。type 就是创建类对象的类。

可以看到，上面的所有东西，也就是所有对象都是通过类来创建的，那么我们可能会好奇，`__class__` 的 `__class__` 会是什么呢？换个说法就是，创建这些类的类是什么呢？
``` python 
print(age.__class__.__class__)
print(name.__class__.__class__)
print(fu.__class__.__class__)
print(mEat.__class__.__class__)
```
输出结果：
``` xml
<class 'type'>
<class 'type'>
<class 'type'>
<class 'type'>
```

可以看出，把他们类的类打印结果。发现打印出来的 class 都是 type 。

一开始也提到了，元类就是类的类。也就是元类就是负责创建类的一种东西。你也可以理解为，元类就是负责生成类的。而 type 就是内建的元类。也就是 Python 自带的元类。

#### 3.3.4 自定义元类

连接起来就是：先定义 metaclass，就可以创建类，最后创建实例。

所以，metaclass 允许你创建类或者修改类。换句话说，你可以把类看成是 metaclass 创建出来的“实例”。

``` python 
class MyObject(object):
    __metaclass__ = something…
[…]
```
如果是这样写的话，Python 就会用元类来创建类 MyObject。当你写下 `class MyObject(object)`，但是类对象 MyObject 还没有在内存中创建。Python 会在类的定义中寻找 `__metaclass__` 属性，如果找到了，Python 就会用它来创建类 MyObject，如果没有找到，就会用内建的 type 函数来创建这个类。如果还不怎么理解，看下下面的流程图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-9-15-38258673.jpg)

举个实例：
``` python 
class Foo(Bar):
    pass
```
它的流程是怎样的呢？
> 首先判断 Foo 中是否有 `__metaclass__` 这个属性？如果有，Python 会在内存中通过 `__metaclass__` 创建一个名字为 Foo 的类对象（注意，这里是类对象）。如果 Python 没有找到`__metaclass__` ，它会继续在 Bar（父类）中寻找`__metaclass__` 属性，并尝试做和前面同样的操作。如果 Python在任何父类中都找不到 `__metaclass__ `，它就会在模块层次中去寻找` __metaclass__` ，并尝试做同样的操作。如果还是找不到`__metaclass__` ,Python 就会用内置的 type 来创建这个类对象。

其实 `__metaclass__` 就是定义了 class 的行为。类似于 class 定义了 instance 的行为，metaclass 则定义了 class 的行为。可以说，class 是 metaclass 的 instance。

现在，我们基本了解了 `__metaclass__` 属性，但是，也没讲过如何使用这个属性，或者说这个属性可以放些什么？

答案就是：可以创建一个类的东西。那么什么可以用来创建一个类呢？type，或者任何使用到 type 或者子类化 type 的东东都可以。

#### 3.4.5 元类的作用

元类的主要目的就是为了当创建类时能够自动地改变类。通常，你会为 API 做这样的事情，你希望可以创建符合当前上下文的类。

假想一个很傻的例子，你决定在你的模块里所有的类的属性都应该是大写形式。有好几种方法可以办到，但其中一种就是通过在模块级别设定`__metaclass__` 。采用这种方法，这个模块中的所有类都会通过这个元类来创建，我们只需要告诉元类把所有的属性都改成大写形式就万事大吉了。


总结：**Python 中的一切都是对象，它们要么是类的实例，要么是元类的实例，除了 type。type 实际上是它自己的元类，在纯 Python 环境中这可不是你能够做到的，这是通过在实现层面耍一些小手段做到的。** 



## 四、线程与进程

线程和进程的概念我就不多赘述了。可自行网上搜索查找资料了解下。

直接看问题：在 Python 中我们要同时执行多个任务怎么办？

有两种解决方案：

1. 一种是启动多个进程，每个进程虽然只有一个线程，但多个进程可以一块执行多个任务。
2. 还有一种方法是启动一个进程，在一个进程内启动多个线程，这样，多个线程也可以一块执行多个任务。

当然还有第三种方法，就是启动多个进程，每个进程再启动多个线程，这样同时执行的任务就更多了，当然这种模型更复杂，实际很少采用。

总结一下就是，多任务的实现有3种方式：

- 多进程模式；
- 多线程模式；
- 多进程+多线程模式。

同时执行多个任务通常各个任务之间并不是没有关联的，而是需要相互通信和协调，有时，任务 1 必须暂停等待任务 2 完成后才能继续执行，有时，任务 3 和任务 4 又不能同时执行，所以，多进程和多线程的程序的复杂度要远远高于我们前面写的单进程单线程的程序。

### 4.1 多线程编程

其实创建线程之后，线程并不是始终保持一个状态的，其状态大概如下：

- New 创建
- Runnable 就绪。等待调度
- Running 运行
- Blocked 阻塞。阻塞可能在 Wait Locked Sleeping
- Dead 消亡

线程有着不同的状态，也有不同的类型。大致可分为：

- 主线程
- 子线程
- 守护线程（后台线程）
- 前台线程

**线程的创建：**

Python 提供两个模块进行多线程的操作，分别是 `thread` 和 `threading`

前者是比较低级的模块，用于更底层的操作，一般应用级别的开发不常用。

``` python 
import time
import threading


class MyThread(threading.Thread):
    def run(self):
        for i in range(5):
            print('thread {}, @number: {}'.format(self.name, i))
            time.sleep(1)

def main():
    print("Start main threading")

    # 创建三个线程
    threads = [MyThread() for i in range(3)]
    # 启动三个线程
    for t in threads:
        t.start()

    print("End Main threading")


if __name__ == '__main__':
    main()
```

这块的内容还有很多，由于该文重点还是为讲解 Python 的基础知识。线程和进程的内容更多还是到网上搜索资料学习，亦或是日后有时间我再更新于此。



## 五、Python 正则表达式

>正则表达式是一个特殊的字符序列，用于判断一个字符串是否与我们所设定的字符序列是否匹配，也就是说检查一个字符串是否与某种模式匹配。

Python 自 1.5 版本起增加了 re 模块，它提供 Perl 风格的正则表达式模式。re 模块使 Python 语言拥有全部的正则表达式功能。

如下代码：
``` python 
# 设定一个常量
a = '学习Python不难'

# 判断是否有 “Python” 这个字符串，使用 PY 自带函数

print('是否含有“Python”这个字符串：{0}'.format(a.index('Python') > -1))
print('是否含有“Python”这个字符串：{0}'.format('Python' in a))
```
输出结果：
``` xml
是否含有“Python”这个字符串：True
是否含有“Python”这个字符串：True
```

上面用 Python 自带函数就能解决的问题，我们就没必要使用正则表达式了，这样做多此一举。

直接举个 Python 中正则表达式使用例子好了：找出字符串中的所有小写字母。

首先我们在 findall 函数中第一个参数写正则表达式的规则，其中`[a-z]`就是匹配任何小写字母，第二个参数只要填写要匹配的字符串就行了。具体如下：

``` python 
import re

# 设定一个常量
a = '学习Python不难'

# 选择 a 里面的所有小写英文字母

re_findall = re.findall('[a-z]', a)

print(re_findall)
```
输出结果：
``` xml
['y', 't', 'h', 'o', 'n']
```
这样我们就拿到了字符串中的所有小写字母了。

补充：

> - 贪婪模式：它的特性是一次性地读入整个字符串，如果不匹配就吐掉最右边的一个字符再匹配，直到找到匹配的字符串或字符串的长度为 0 为止。它的宗旨是读尽可能多的字符，所以当读到第一个匹配时就立刻返回。
> - 懒惰模式：它的特性是从字符串的左边开始，试图不读入字符串中的字符进行匹配，失败，则多读一个字符，再匹配，如此循环，当找到一个匹配时会返回该匹配的字符串，然后再次进行匹配直到字符串结束。

关于正则表达式的更多的学习还是找网上资料看看吧。



## 六、闭包

通过解决一个需求问题来了解闭包。

> 这个需求是这样的，我们需要一直记录自己的学习时间，以分钟为单位。就好比我学习了 2 分钟，就返回 2 ，然后隔了一阵子，我学习了 10 分钟，那么就返回 12 ，像这样把学习时间一直累加下去。

面对这个需求，我们一般都会创建一个全局变量来记录时间，然后用一个方法来新增每次的学习时间，通常都会写成下面这个形式：
``` python
time = 0

def insert_time(min):
    time = time + min
    return  time

print(insert_time(2))
print(insert_time(10))
```
其实，这个在 Python 里面是会报错的。会报如下错误：`UnboundLocalError: local variable 'time' referenced before assignment`

那是因为，在 Python 中，**如果一个函数使用了和全局变量相同的名字且改变了该变量的值，那么该变量就会变成局部变量**，那么就会造成在函数中我们没有进行定义就引用了，所以会报该错误。

我们可以使用 global 关键字，具体修改如下：
``` python 
time = 0

def insert_time(min):
    global  time
    time = time + min
    return  time

print(insert_time(2))
print(insert_time(10))
```
输出结果如下：
``` xml
2
12
```
可是啊，这里使用了全局变量，我们在开发中能尽量避免使用全局变量的就尽量避免使用。因为不同模块，不同函数都可以自由的访问全局变量，可能会造成全局变量的不可预知性。比如程序员甲修改了全局变量 time 的值，然后程序员乙同时也对 time 进行了修改，如果其中有错误，这种错误是很难发现和更正的。

这时候我们使用闭包来解决一下，先直接看代码：
``` python 
time = 0

def study_time(time):
    def insert_time(min):
        nonlocal  time
        time = time + min
        return time

    return insert_time

f = study_time(time)
print(f(2))
print(time)
print(f(10))
print(time)
```
输出结果如下：
``` xml
2
0
12
0
```
这里最直接的表现就是全局变量 time 至此至终都没有修改过，这里还是用了 nonlocal 关键字，表示在函数或其他作用域中使用外层（非全局）变量。那么上面那段代码具体的运行流程是怎样的。我们可以看下下图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-9-15-76967350.jpg)

**这种内部函数的局部作用域中可以访问外部函数局部作用域中变量的行为，我们称为： 闭包。** 更加直接的表达方式就是，当某个函数被当成对象返回时，夹带了外部变量，就形成了一个闭包。

有没有什么办法来验证一下这个函数就是闭包呢？

有的，所有函数都有一个 `__closure__` 属性，如果函数是闭包的话，那么它返回的是一个由 cell 组成的元组对象。cell 对象的 cell_contents 属性就是存储在闭包中的变量。看代码：
``` python 
ime = 0


def study_time(time):
    def insert_time(min):
        nonlocal  time
        time = time + min
        return time

    return insert_time


f = study_time(time)
print(f.__closure__)
print(f(2))
print(time)
print(f.__closure__[0].cell_contents)
print(f(10))
print(time)
print(f.__closure__[0].cell_contents)
```
打印结果为：
``` xml
(<cell at 0x0000000000410C48: int object at 0x000000001D6AB420>,)
2
0
2
12
0
12
```
从打印结果可见，传进来的值一直存储在闭包的 cell_contents 中,因此，这也就是闭包的最大特点，可以将父函数的变量与其内部定义的函数绑定。就算生成闭包的父函数已经释放了，闭包仍然存在。

闭包的过程其实好比类（父函数）生成实例（闭包），不同的是父函数只在调用时执行，执行完毕后其环境就会释放，而类则在文件执行时创建，一般程序执行完毕后作用域才释放，因此对一些需要重用的功能且不足以定义为类的行为，使用闭包会比使用类占用更少的资源，且更轻巧灵活。



## 七、装饰器

### 7.1 什么是装饰器

通过一个需求，一步一步来了解 Python 装饰器。首先有这么一个输出员工打卡信息的函数：

``` python 
def punch():
    print('昵称：小明  部门：研发部 上班打卡成功')

punch()
```
输出的结果：
``` python 
昵称：小明  部门：研发部 上班打卡成功
```
然后，产品反馈，不行啊，怎么上班打卡没有具体的日期，加上打卡的具体日期吧，这应该很简单，分分钟解决啦。好吧，那就直接添加打印日期的代码吧，如下：
``` python 
import time

def punch():
    print(time.strftime('%Y-%m-%d', time.localtime(time.time())))
    print('昵称：小明  部门：研发部 上班打卡成功')

punch()
```
输出的结果：
``` xml
2018-01-09
昵称：小明  部门：研发部 上班打卡成功
```
这样改是可以，可是这样改是改变了函数的功能结构的，本身这个函数定义的时候就是打印某个员工的信息和提示打卡成功，现在增加打印日期的代码，可能会造成很多代码重复的问题。比如，还有一个地方只需要打印员工信息和打卡成功就行了，不需要日期，那么你又要重写一个函数吗？而且打印当前日期的这个功能方法是经常使用的，是可以作为公共函数给各个模块方法调用的。当然，这都是作为一个整体项目来考虑的。

既然是这样，我们可以使用函数式编程来修改这部分的代码。因为通过之前的学习，我们知道 Python 函数有两个特点，函数也是一个对象，而且函数里可以嵌套函数，那么修改一下代码变成下面这个样子：
``` python 
import time

def punch():
    print('昵称：小明  部门：研发部 上班打卡成功')

def add_time(func):
    print(time.strftime('%Y-%m-%d', time.localtime(time.time())))
    func()

add_time(punch)
```
输出的结果：
``` xml
2018-01-09
昵称：小明  部门：研发部 上班打卡成功
```
这样是不是发现，这样子就没有改动 `punch` 方法，而且任何需要用到打印当前日期的函数都可以把函数传进 add_time 就可以了。

使用函数编程是不是很方便，但是，我们每次调用的时候，我们都不得不把原来的函数作为参数传递进去，还能不能有更好的实现方式呢？有的，就是本文要介绍的装饰器，因为装饰器的写法其实跟闭包是差不多的，不过没有了自由变量，那么这里直接给出上面那段代码的装饰器写法，来对比一下，装饰器的写法和函数式编程有啥不同。

``` python 
import time

def decorator(func):
    def punch():
        print(time.strftime('%Y-%m-%d', time.localtime(time.time())))
        func()

    return punch

def punch():
    print('昵称：小明  部门：研发部 上班打卡成功')

f = decorator(punch)
f()
```
输出的结果：
``` xml
2018-01-09
昵称：小明  部门：研发部 上班打卡成功
```
通过代码，能知道装饰器函数一般做这三件事：

1. 接收一个函数作为参数
2. 嵌套一个包装函数, 包装函数会接收原函数的相同参数，并执行原函数，且还会执行附加功能
3. 返回嵌套函数

### 7.2 语法糖 

我们看上面的代码可以知道， Python 在引入装饰器 （Decorator） 的时候，没有引入任何新的语法特性，都是基于函数的语法特性。这也就说明了装饰器不是 Python 特有的，而是每个语言通用的一种编程思想。只不过 Python 设计出了 `@` 语法糖，**让定义装饰器，把装饰器调用原函数再把结果赋值为原函数的对象名的过程**变得更加简单，方便，易操作，所以 Python 装饰器的核心可以说就是它的语法糖。

那么怎么使用它的语法糖呢？很简单，根据上面的写法写完装饰器函数后，直接在原来的函数上加 `@` 和装饰器的函数名。如下：
``` python 
import time

def decorator(func):
    def punch():
        print(time.strftime('%Y-%m-%d', time.localtime(time.time())))
        func()

    return punch

@decorator
def punch():
    print('昵称：小明  部门：研发部 上班打卡成功')

punch()
```
输出结果：
``` xml
2018-01-09
昵称：小明  部门：研发部 上班打卡成功
```
那么这就很方便了，方便在我们的调用上，比如例子中的，使用了装饰器后，直接在原本的函数上加上装饰器的语法糖就可以了，本函数也无虚任何改变，调用的地方也不需修改。

不过这里一直有个问题，就是输出打卡信息的是固定的，那么我们需要通过参数来传递，装饰器该怎么写呢？装饰器中的函数可以使用 `*args` 可变参数，可是仅仅使用 `*args` 是不能完全包括所有参数的情况，比如关键字参数就不能了，为了能兼容关键字参数，我们还需要加上 `**kwargs` 。

因此，装饰器的最终形式可以写成这样：
``` python
import time

def decorator(func):
    def punch(*args, **kwargs):
        print(time.strftime('%Y-%m-%d', time.localtime(time.time())))
        func(*args, **kwargs)

    return punch

 
@decorator
def punch(name, department):
    print('昵称：{0}  部门：{1} 上班打卡成功'.format(name, department))

@decorator
def print_args(reason, **kwargs):
    print(reason)
    print(kwargs)

punch('小明', '研发部')
print_args('小明', sex='男', age=99)
```
输出的结果：
``` xml
2018-01-09
昵称：小明  部门：研发部 上班打卡成功
2018-01-09
小明
{'sex': '男', 'age': 99}
```



本文内容大部分来源：

- 两点水：[《草根学Python》](https://juejin.im/collection/5947e6851e35c9353d939136)


