## 1. 认识 Anaconda

### 1.1 Anaconda 安装

首先有必要对 Anaconda、conda、pip、virtualenv 等这些有个认识，以及 Anaconda 在 Windows、MacOS、Linux 下的安装教程，参考：

- [Anaconda介绍、安装及使用教程](https://zhuanlan.zhihu.com/p/32925500)
- [Python，Pycharm，Anaconda等的关系与安装过程~为初学者跳过各种坑](https://www.cnblogs.com/tq007/p/7281105.html)
- [致Python初学者：Anaconda入门使用指南](http://python.jobbole.com/87522/)
- [Windows下Anaconda的安装和简单使用](https://blog.csdn.net/DQ_DM/article/details/47065323)

简单总结下：

- Anaconda 就是可以便捷获取包且对包能够进行管理，同时对环境可以统一管理的发行版本。Anaconda 包含了 conda、Python 在内的超过 180 个科学包及其依赖项。简单说，当在电脑上安装好 Anaconda 以后，就相当于安装好了 Python，还有一些常用的库，如 numpy，scrip，matplotlib 等库。
- conda 是包及其依赖项和环境的管理工具。
- pip 是用于安装和管理软件包的包管理器。
- virtualenv 是用于创建一个独立的 Python 环境的工具。

安装 Anaconda ：点击 [这里](https://www.anaconda.com/distribution/) 下载对应自己想要安装的 python 版本的 Anaconda ，然后下一步下一步安装就行。

> 注意：因为 anaconda 是自带 Python 的，所以不需要自己再去下载安装 Python 了，当然，如果你已经安装了 Python 也不要紧，不会发生冲突的。
>
> 这里提下，下载的 Anaconda 和 Python 版本对应情况，图片来源网络：
>
> ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190723213904.png)
>
> 解释一下上表，anaconda 在每次发布新版本的时候都会给 python3 和 python2 都发布一个包，版本号是一样的。假设你想安装 `python3.6.5`，就去下载 `anaconda3-5.2.0`；假设你想安装 `python2.7.14`，就去下载 `anaconda2-5.2.0`。

在安装过程有遇到的问题和注意的地方，在这记录下：

（1）我的电脑就已经安装过 Python 环境并设置到了电脑的用户变量 PATH 中，但在安装 Anaconda 完毕之后，我的电脑只有这么一个菜单，如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/18-7-12-35379422.jpg)


不知道怎么回事，尝试卸载了再重装几次还是这样。然后按照 [Python3学习壹——Anaconda+Pycharm环境搭建](https://cdn2.jianshu.io/p/be30a6b15371?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation) 作者的解决方式，删除了之前的 Python 环境变量再重装，但也不行，以及按照该文 [关于安装Anaconda3各种各样的问题，吐血总结](https://blog.csdn.net/qq_36015370/article/details/79484455/) 的方式，进入 Anaconda 安装目录下的 CMD 中敲 `python .\Lib\_nsis.py mkmenus` 也没成。无奈之下，干脆卸载了 Python 再安装 Anaconda，也没解决这个问题。

最后，尝试把安装目录改为别的地方，竟然 ok 了。（注：之前一直选择的是和 Python安装目录的同一目录，莫非这个导致的？）

（2）假设 Anaconda 安装目录为 `D:\Anaconda3` ，则把 `D:\Anaconda3;`、 `D:\Anaconda3\Scripts;`、 `D:\Anaconda3\Library\bin;`，把它们添加到 PATH 中。

### 1.2 源更改

使用 Anaconda 管理，安装 Python 库的时候默认是使用国外的源，这时候下载速度会很慢，国内的源下载速度要好很多。参考：

- [在pycharm中配置Anaconda以及pip源配置](https://blog.csdn.net/u012513525/article/details/54947398)
- [pip和conda安装源更改](https://blog.csdn.net/sxf1061926959/article/details/54091748)

#### (1) pip 源更改

配置环境：Windows7 （64位），Python3.6

（1）临时使用

pip 后加参数：`-i https://pypi.tuna.tsinghua.edu.cn/simple`，例如：

``` xml
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas 
```

（2）永久使用

Linux 下：

修改 ~/.pip/pip.conf (没有就创建一个)， 修改 index-url至tuna，内容如下：

``` xml
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

Windows 下：

1. 在 Windows 文件管理器中，输入 `%APPDATA%` 回车

2. 接着会定位到一个新的目录，在这个目录中新建一个 pip 文件夹，然后在 pip 文件夹中新建个 pip.ini 文件

3. 最后再新建的 pip.ini 文件中输入一下内容：

   ``` xml
   [global]
   index-url = https://pypi.tuna.tsinghua.edu.cn/simple
   ```

#### (2) conda 源更改

在安装了 Anaconda 后，我们也可以使用 Anaconda 来进行 Python 库的安装，同样的也需要进行源的配置。

这个配置方法就很简单了，你只需要在配置了 Anaconda 的终端（Terminal）输入一下命令即可：

``` xml
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

此时，目录 `C:\Users<你的用户名>` 下就会生成配置文件 `.condarc`，内容如下：

``` xml
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true
```

好了，源的更改到此就完成了。大家有兴趣可以去清华大学的 [开源镜像站](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) 看看，会有很多惊喜的。

查看当前配置信息 `conda info`，容如下，即修改成功，关注 channel URLs 字段内容：

``` xml
 			   platform : win-32
          conda version : 4.3.22
       conda is private : False
      conda-env version : 4.3.22
       requests version : 2.12.4
           channel URLs : https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/win-32
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/noarch
```



## 2. conda 的包管理 

Anaconda 为我们提供方便的包管理命令：conda，下面我们来看看都有哪些有用的命令。

``` xml
conda --v	#验证conda已被安装
conda --h	#查看conda帮助信息

conda list	# 查看已经安装的packages
conda list -n py3	# 查看某个指定环境py3的已安装包

conda search numpy	# 查找package信息
conda search --full-name python	#查找全名为“python”的包有哪些版本可供安装。

conda install -n py3 numpy	# 安装package到名为py3环境,如果不用-n指定环境名称，则被安装在当前活跃环境
conda install scipy	# 安装scipy到当前活跃环境
conda update --all 或者 conda upgrade --all	# 更新所有包
conda update -n py3 numpy	# 更新名为py3环境的package
conda remove -n py3 numpy	# 删除名为py3环境的package
```

由于 conda 将 conda、python 等都视为 package，因此，完全可以使用 conda 来管理 conda 和 python 的版本，例如：

``` xml
conda update conda # 更新conda，保持conda最新
conda update anaconda # 更新anaconda
conda update python # 更新python
```

卸载 Anaconda：

``` xml
Linux 或 MacOS 下： rm -rf ~/anaconda3
Windows 下：控制面板 → 卸载程序 → 选择“Python X.X (Anaconda)” → 点击“删除程序”
```

当使用 `conda install` 无法进行安装时，可以使用 `pip` 进行安装，例如：see 包。

```
pip install <package_name>

注1： <package_name> 为指定安装包的名称。包名两边不加尖括号“<>”。
如： pip install see 即安装see包。

注2：当前是哪个环境，pip 会把包安装在当前环境。

1、pip只是包管理器，无法对环境进行管理。因此如果想在指定环境中使用pip进行安装包，则需要先切换到指定环境中，再使用pip命令安装包。
2、pip无法更新python，因为pip并不将python视为包。
3、pip可以安装一些conda无法安装的包；conda也可以安装一些pip无法安装的包。因此当使用一种命令无法安装包时，可以尝试用另一种命令。
```

另：当使用 `conda install` 无法进行安装时，可以考虑从 http://Anaconda.org 搜索要安装的包名，并获取安装包的命令，进行安装。



## 3. Anaconda 环境管理

关于 Anaconda 的环境重点说下。

**（1）创建新环境：** `conda create --name <env_name> <package_names>` 

``` xml
<env_name> 即创建的环境名。建议以英文命名，且不加空格，名称两边不加尖括号“<>”。
<package_names> 即安装在环境中的包名。名称两边不加尖括号“<>”。

① 如果要安装指定的版本号，则只需要在包名后面以 = 和版本号的形式执行。如：conda create --name py2 python=2.7 ，即创建一个名为“py2”的环境，环境中安装版本为2.7的python。

② 如果要在新创建的环境中创建多个包，则直接在 <package_names> 后以空格隔开，添加多个包名即可。如： conda create -n py3 python=3.5 numpy pandas ，即创建一个名为“python3”的环境，环境中安装版本为3.5的python，同时也安装了numpy和pandas。

注1：--name 可以替换为 -n 。
注2：默认情况下，新创建的环境将会被保存在 Anaconda 安装目录下的 envs 目录，其中， <user_name> 为当前用户的用户名。
```

注：假设创建的新环境为 py3（会在 Anaconda 安装目录下 env 文件夹下生产同名的文件夹 py3），`activate py3` 进入 py3 环境中，再运行命令：`conda install tensorflow`，则会安装 `tensorflow` 到 py3 此环境中，即下载的 TensorFlow 文件，存放在 `D:\Anaconda3\envs\py3\Lib\site-packages` 下。

**（2）切换环境**

① Linux 或 macOS 下：`source activate <env_name>`

② Windows 下：`activate <env_name>`

③ 提示

- 如果创建环境后安装Python时没有指定Python的版本，那么将会安装与Anaconda版本相同的Python版本，即如果安装Anaconda第2版，则会自动安装Python 2.x；如果安装Anaconda第3版，则会自动安装Python 3.x。
- 当成功切换环境之后，在该行行首将以“(env_name)”或“[env_name]”开头。其中，“env_name”为切换到的环境名。如：在macOS系统中执行 `source active python2`，即切换至名为“python2”的环境，则行首将会以(python2)开头。

**（3）退出环境至root**

① Linux 或 macOS 下：`source deactivate`

② Windows 下：`deactivate`

③ 提示：当执行退出当前环境，回到root环境命令后，原本行首以“(env_name)”或“[env_name]”开头的字符将不再显示。

**（4）显示已创建环境**：`conda info --envs` 或 `conda env list`

**（5）复制环境**：`conda create --name <new_env_name> --clone <copied_env_name>`

``` xml
注意：
① <copied_env_name> 即为被复制/克隆环境名。环境名两边不加尖括号“<>”。

② <new_env_name> 即为复制之后新环境的名称。环境名两边不加尖括号“<>”。

③ conda create --name py2 --clone python2 ，即为克隆名为“python2”的环境，克隆后的新环境名为“py2”。此时，环境中将同时存在“python2”和“py2”环境，且两个环境的配置相同。
```

**（6） 删除环境**：`conda remove --name <env_name> --all`

- 注意： `<env_name>` 为被删除环境的名称。环境名两边不加尖括号“<>”。

有了虚拟环境，在 PyCharm IDE 下就可以指定虚拟环境下的解析器，非常方便。具体做法是：File --> Default settings --> Default project --> project interpreter，接着点击 project interpreter 的右边的小齿轮，选择 add local ，选择 anaconda 文件路径下想要使用的虚拟环境下的 python.exe，接着 PyCharm 会更新解释器，导入模块等，要稍等一点时间。

注：在 PyCharm IDE 中安装库会自动下载到指定解析器所在的环境的`<env-name>\Lib\site-packages` 下。

另外：关于创建虚拟环境，如果你只是安装了 Python，官方自带创建虚拟环境功能，如创建名为 myenv 的虚拟环境  `python -m venv myenv `；另外也可以通过安装 `virtualenv` 来创建虚拟环境，步骤如下：

1. 首先安装 `virtualenv`：`pip install virtualenv`

2. 创建虚拟环境：

   ``` xml
   $ mkdir myproject
   $ cd myproject
   $ virtualenv venv
   ```

   创建了一个名为 `myproject` 的文件夹，然后这里边创建虚拟环境 `venv`。参考：[Anacodna之conda VS Virtualenv VS Python 3 venv 对比使用教程，创建虚拟环境](https://segmentfault.com/a/1190000005828284)

关于 `pip` 和 `virtualenv` 以及 `conda` 区别，可以这么理解：`pip` 是一个包管理器，`virtualenv` 是一个环境管理器，而 `conda` 就是它们俩的综合体。

## 4. 注意和总结

**(1) 注意**

在安装有多个 Python 环境情况下，比如除了单独的 Python，还有 Anaconda 环境，那么在 Windows 的 CMD 下使用 `pip install <package>`、`conda install <package>` 等命令到底使用的哪个环境呢？毕竟还是要清楚自己安装的库到底是安装在哪吧。

（1）查看 pip 命令来源哪个环境：`pip --version` 或 `pip -V`，可以看到如下：

``` xml
pip 9.0.1 from D: devInstall\ Anaconda3\ lib\ site-packages <python 3.6)
```

如上表示来自 Anaconda 环境下的 pip，则使用 pip 安装的库安装在对应的 anaconda 环境下。

（2）使用：`conda install <package>` 表示安装在默认的 anaconda 环境下，如果想要安装到新建的虚拟环境，如 `py3` ，则先切换到该虚拟环境 `activate <env_name>`（windows 下），然后使用 conda install 安装即可。

（3）查看 CMD 下使用的 python 命令来源哪个环境，`python --version` 或 `python -V`，比如可以看到：

``` xml
Python 3.6.4:: Anaconda, Inc.
```

表明来源 Anaconda 环境。





