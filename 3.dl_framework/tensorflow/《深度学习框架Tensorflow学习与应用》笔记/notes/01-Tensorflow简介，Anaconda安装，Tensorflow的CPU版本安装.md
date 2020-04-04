## 1. Anaconda安装

1）Window、MacOS、Linux 都已支持 Tensorflow。

2）Window 用户只能使用 python3.5(64bit)。MacOS、Linux 支持 python2.7 和 python3.3+。

3）有 GPU 可以安装带 GPU 版本的，没有 GPU 就安装 CPU 版本的。

4） 推荐安装 Anaconda，pip 版本大于 8.1。

在学习过程中，建议使用 Jupyter Notebook 编程（当然也可以用其他工具，如 PyCharm），因安装完 Anaconda 自带 Jupyter Notebook，可以找到直接打开即可开始。

打开之后会打开默认浏览器，地址 `http://localhost:8888`，然后就可以在浏览器下面新建文件进行代码编写等操作了，其中，默认保存的路径为 C 盘用户文件夹下，如：`C:\Users\用户名`。我们可以修改路径为我们自己的想要的目录之下，修改操作：

1. 打开 cmd 输入命令 `jupyter notebook --generate-config`，可以看到生成文件的路径（可以看到默认在 `C:\Users\用户名\.jupyter` 文件夹下），这个就是生成的配置文件 `jupyter_notebook_config.py`

   ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-10-8-14115072.jpg)

2. 然后打开这个配置文件，找到 `#c.NotebookApp.notebook_dir = ' '`，把它改成：

   ``` xml
   c.NotebookApp.notebook_dir = '你想要设置的路径'
   ```

   如：`c.NotebookApp.notebook_dir = 'D:/Python/jupyter'`，那么以后再 Jupyter 上保存的文件就在`D:/Python/jupyter`文件夹里了。

   注1：如果想要检测是否修改成功，可以在你设置的目录下里添加一个文件夹，例如：我在 jupyter 文件夹里添加了文件夹 text，那么在 Jupyter（浏览器上）就可以看到该文件夹了，则表明修改路径成功！

   注2：在进行如上修改之后，我发现未成功。网上找到了篇文章（[Anaconda Jupyter默认路径及修改无效解决方案](https://blog.csdn.net/mirrorui_/article/details/80605613)）也是遇到同样问题，按照文章解决方法，最后修改路径总算成功。其解决方法如下：

   1. 找到 Jupyter 快捷方式，右键属性，并修改起始位置为你设置的路径，即刚才的「你想要设置的路径」
   2. 在 Jupyter 的快捷方式属性中，有栏叫 `目标`，将这栏最后的 `%USERPROFILE%` 去掉。

另外：如果需要检测 TensorFlow 是否安装成功，可以打开 Jupyter 新建 python 文件，输入 `import tensorflow as tf` 运行，看是否报错。



## 2. TensorFlow安装

1、Windows安装TensorFlow

- CPU版本，管理员方式打开命令提示符，输入命令：`pip install tensorflow`
- GPU版本，管理员方式打开命令提示符，输入命令：`pip install tensorflow-gpu`

更新 TensorFlow：

``` python
pip uninstall tensorflow
pip install tensorflow
```

注意，如果在安装过程中提示需要 `MSVCP140.DLL`，则下载安装：

``` xml
TensorFlow requires MSVCP140.DLL, which may not be installed on your system. If,
when you import tensorflow as tf, you see an error about No module named
"_pywrap_tensorflow" and/or DLL load failed, check whether MSVCP140.DLL is in
your %PATH% and, if not, you should install the Visual C++ 2015 redistributable (x64
version).
```

2、Linux和MacOS安装Tensorflow

- CPU版本
  - Python 2.7用户：`pip install tensorflow`
  - Python3.3+用户：`pip3 install tensorflow`
- GPU版本
  - Python 2.7用户：`pip install tensorflow-gpu`
  - Python3.3+用户：`pip3 install tensorflow-gpu`

关于 pip 和 pip3 的区别：

- [python3中的pip和pip3](https://segmentfault.com/q/1010000010354189)
- [安装python3后使用pip和pip3的区别](https://zhidao.baidu.com/question/494182519781589612.html?qbl=relate_question_1)

> `pip`和`pip3`都在`Python36\Scripts\`目录下，如果同时装有 python2 和 python3，pip 默认给`python2` 用，`pip3`指定给 python3 用。如果只装有 python3，则`pip`和`pip3`是等价的。

> 1、其实这两个命令效果是一样的，没有区别：
>
> （1）比如安装库 numpy，pip3  install  numpy 或者 pip  install  numpy：只是当一台电脑同时有多个版本的 Python 的时候，用 pip3 就可以自动区分用 Python3 来安装库。是为了避免和 Python2 发生冲突的。
>
> （2）如果你的电脑只安装了 Python3，那么不管用 pip 还是 pip3 都一样的。
>
> 2、安装了 python3 之后，会有 pip3
>
> （1）使用 pip install XXX ：
>
> 新安装的库会放在这个目录下面：python2.7/site-packages；
>
> （2）使用 pip3 install XXX ：
>
> 新安装的库会放在这个目录下面：python3.6/site-packages；
>
> （3）如果使用 python3 执行程序，那么就不能 import python2.7/site-packages 中的库。



