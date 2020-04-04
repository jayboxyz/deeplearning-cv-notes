# 如何一键安装所有第三方库文件？

## pip freeze



在查看别人的 Python 项目时，经常会看到一个 `requirements.txt` 文件，里面记录了当前程序的所有依赖包及其精确版本号。这个文件有点类似与 Rails 的 Gemfile。其作用是用来在另一台 PC 上重新构建项目所需要的运行环境依赖。

`requirements.txt` 用来记录项目所有的依赖包和版本号，只需要一个简单的 pip 命令就能完成。

进入到需要导出所有 Python 库的那个环境，然后使用那个环境下的 pip ：

``` python
pip freeze > requirements.txt
```

> requirement.txt 文件默认输出在桌面。
>
> 注：查看源文件，pip 的 freeze 命令用于生成将当前项目的 pip 类库列表生成 requirements.txt 文件。

然后就可以用：

``` python
pip install -r requirements.txt
```

来一次性安装 `requirements.txt` 里面所有的依赖包，真是非常方便。



`requirements.txt` 文件类似如下：

``` python
Django=1.3.1
South>=0.7
django-debug-toolbar
```

将模块放在一个列表中，每一行只有一项。

---



## pipreqs

### pipreqs 的作用

> 一起开发项目的时候总是要搭建环境和部署环境的，这个时候必须得有个 python 第三方包的 list，一般都叫做requirements.txt。 如果一个项目使用时 virtualenv 环境，还好办 `pip freeze` 就可以解决，但是如果一个项目的依赖 list 没有维护，而且又是环境混用，那就不好整理的呀，不过，这里安利一个工具 pipreqs，可以自动根据源码生成 `requirements.txt`。

pip freeze 命令：

``` python
pip freeze > requirements.txt
```

这种方式配合 virtualenv 才好使，否则把整个环境中的包都列出来了。



### pipreqs 的使用

pipreqs  这个工具的好处是可以通过对项目目录的扫描，自动发现使用了那些类库，自动生成依赖清单。缺点是可能会有些偏差，需要检查并自己调整下。

pipreqs 的安装：`pip install pipreqs`

使用方式也比较简单，直接进入项目下然后使用 `pipreqs ./` 命令即可，如：

```
pipreqs ./
```

如果是 Windows 系统，会报编码错误 (UnicodeDecodeError: 'gbk' codec can't decode byte 0xa8 in position 24: illegal multibyte sequence)  。这是由于编码问题所导致的，加上 encoding 参数即可，如下：

``` python
pipreqs ./ --encoding=utf-8
```

生成 `requirements.txt` 文件后，可以根据这个文件下载所有的依赖。

``` python
pip install -r requriements.txt
```

附：

``` xml
详细用法：
pipreqs [options] <path>

选项：
    --use-local仅使用本地包信息而不是查询PyPI
    --pypi-server <url>使用自定义PyPi服务器
    --proxy <url>使用Proxy，参数将传递给请求库。你也可以设置
    
    终端中的环境参数：
    $ export HTTP_PROXY =“http://10.10.1.10:3128”
    $ export HTTPS_PROXY =“https://10.10.1.10:1080”
    --debug打印调试信息
    --ignore <dirs> ...忽略额外的目录
    --encoding <charset>使用编码参数打开文件
    --savepath <file>保存给定文件中的需求列表
    --print输出标准输出中的需求列表
    --force覆盖现有的requirements.txt
    --diff <file>将requirements.txt中的模块与项目导入进行比较。
    --clean <file>通过删除未在项目中导入的模块来清理requirements.txt。
```





## 参考文章：

- [python 批量导出项目所依赖的所有库文件及安装的方法（包导出与导入）](<https://blog.csdn.net/mezheng/article/details/84317515>)
- [Python使用requirements.txt安装类库](https://www.cnblogs.com/zknublx/p/5953921.html)
- [浅谈pipreqs组件(自动生成需要导入的模块信息)](https://www.cnblogs.com/fu-yong/p/9213723.html)

