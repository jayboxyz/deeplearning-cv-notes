## 一、什么是 Jupyter Notebook？

### 1. 简介

Jupyter Notebook 是基于网页的用于交互计算的应用程序。其可被应用于全过程计算：开发、文档编写、运行代码和展示结果。——from：[Jupyter Notebook官方介绍](<https://jupyter-notebook.readthedocs.io/en/stable/notebook.html)。

简而言之，Jupyter Notebook 是以网页的形式打开，可以在网页页面中**直接编写代码**和**运行代码**，代码的**运行结果**也会直接在代码块下显示的程序。如在编程过程中需要编写说明文档，可在同一个页面中直接编写，便于作及时的说明和解释。

### 2. 组成部分

（1）网页应用

网页应用即基于网页形式的、结合了编写说明文档、数学公式、交互计算和其他富媒体形式的工具。**简言之，网页应用是可以实现各种功能的工具。**

（2）文档

即 Jupyter Notebook 中所有交互计算、编写说明文档、数学公式、图片以及其他富媒体形式的输入和输出，都是以文档的形式体现的。

这些文档是保存为后缀名为 `.ipynb` 的 `JSON` 格式文件，不仅便于版本控制，也方便与他人共享。

此外，文档还可以导出为：HTML、LaTeX、PDF 等格式。

### 3.  Jupyter Notebook 特点

- 编程时具有语法高亮、缩进、tab 补全的功能。
- 可直接通过浏览器运行代码，同时在代码块下方展示运行结果。
- 以富媒体格式展示计算结果。富媒体格式包括：HTML，LaTeX，PNG，SVG 等。
- 对代码编写说明文档或语句时，支持 Markdown 语法。
- 支持使用 LaTeX 编写数学性说明。



## 二、安装 Jupyter Notebook

### 1. 安装

安装 Jupyter Notebook 的前提是需要安装了 Python（3.3版本及以上，或2.7版本）。

**（1）使用Anaconda安装**

如果你是小白或是觉得麻烦，那么建议你通过安装 Anaconda 来解决 Jupyter Notebook 的安装问题，因为 Anaconda 已经自动为你安装了 Jupter Notebook 及其他工具，还有 Python 中超过 180 个科学包及其依赖项。

进入 Anaconda 的 [官方下载页面](<https://www.anaconda.com/distribution/>) 自行选择对应平台 Anaconda 下载安装。如果还不知道什么 Anaconda，建议网上找下资料了解下。Anaconda 的安装我就不多说了，这里贴个链接：<https://zhuanlan.zhihu.com/p/32925500>

安装完 Anaconda 发行版时已经自动为你安装了 Jupyter Notebook 的，但如果没有自动安装，那么就在终端（Linux或macOS的“终端”，Windows的“Anaconda Prompt”，以下均简称“终端”）中输入以下命令安装：

``` python
conda install jupyter notebook
```

**（2）使用 pip 安装**

如果你是有经验的 Python 玩家，可以使用 pip 命令来安装 Jupyter Notebook。下面的命令输入都是在终端当中进行的。

1. 把 pip 升级到最新版本

   - Python 3.x：`pip3 install --upgrade pip`
   - Python 2.x：`pip install --upgrade pip`

   注意：老版本的 pip 在安装 Jupyter Notebook 过程中或面临依赖项无法同步安装的问题。因此强烈建议先把 pip 升级到最新版本。

2. 安装 Jupyter Notebook

   - Python 3.x：`pip3 install jupyter`
   - Python 2.x：`pip install jupyter`



## 三、运行和使用 Jupyter Notebook

### 1. 运行

（1）在默认端口启动：`jupyter notebook`

执行命令之后，在终端中将会显示一系列 notebook 的服务器信息，同时浏览器将会自动启动 Jupyter Notebook。

注意：之后在 Jupyter Notebook 的所有操作，都请保持终端不要关闭，因为一旦关闭终端，就会断开与本地服务器的链接，你将无法在 Jupyter Notebook 中进行其他操作啦。

浏览器地址栏中默认地将会显示：`http://localhost:8888`。其中，“localhost”指的是本机，“8888”则是端口号。

如果你同时启动了多个 Jupyter Notebook，由于默认端口“8888”被占用，因此地址栏中的数字将从“8888”起，每多启动一个Jupyter Notebook数字就加 1，如“8889”、“8890”……

（2）如果你想自定义端口号来启动 Jupyter Notebook，可以在终端中输入以下命令：`jupyter notebook --port <port_number>`

其中，“<port_number>”是自定义端口号，直接以数字的形式写在命令当中，数字两边不加尖括号“<>”。如：`jupyter notebook --port 9999`，即在端口号为“9999”的服务器启动 Jupyter Notebook。

（3）如果你只是想启动 Jupyter Notebook 的服务器但不打算立刻进入到主页面，那么就无需立刻启动浏览器。在终端中输入：`jupyter notebook --no-browser`

此时，将会在终端显示启动的服务器信息，并在服务器启动之后，显示出打开浏览器页面的链接。当你需要启动浏览器页面时，只需要复制链接，并粘贴在浏览器的地址栏中，轻按回车变转到了你的 Jupyter Notebook 页面。

### 2. 使用

当执行完启动命令之后，浏览器将会进入到 Jupyter 的主页面。接下来就可以使用 Jupyter 了。

使用 Jupyter 进行基本的代码编写等工作，网上找下文章看下马上就能玩，相信不难，先暂时不记录了…

（待更新…

以上内容大部分来源：[Jupyter Notebook介绍、安装及使用教程 - 知乎](<https://zhuanlan.zhihu.com/p/33105153>)



## 四、修改 Jupyter Notebook 的默认工作目录

第一次打开 Anaconda 中自带的 Jupyter，默认路径是 `C:\Users\用户名\`，如果不想保存在该目录，可以进行修改，更换到别的目录。

（1）在 Anaconda Prompt 中生成配置文件

打开 Anaconda Prompt，输入如下命令：`jupyter notebook --generate-config`

（2）找到生成的 `jupyter_notebook_config.py` 文件(可以使用 everything 搜索)，文本打开后找到`#c.NotebookApp.notebook_dir = ''` ，将 `#` 注释标记去掉，填入自己的工作路径，如：

``` xml
c.NotebookApp.notebook_dir = u'D:\jupyter-code'
```

保存。（注意：工作路径不能出现中文，否则无法打开 Jupyter Notebook）

上面两步完成之后，打开 Jupyter，若没有成功更改工作目录，继续下面这一步。

（3）修改 JupyterNotebook 快捷方式的目标属性

右击 JupyterNotebook 快捷方式，选择【属性】，删除【目标】属性中的【%USERPROFILE%】，点击【应用】–【确定】。

再次打开 JupyterNotebook 发现工作目录已经修改为我们自己的工作目录。 

——from：[修改Jupyter Notebook的默认工作目录](<https://blog.csdn.net/yuanxiang01/article/details/79217469>)





