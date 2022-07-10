# 一、深度学习硬件选购

- [Keras windows - Keras中文文档](https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_windows/)（含基本开发环境搭建、keras 框架搭建）



# 二、基本认识

1.查看显卡（GPU）是否支持 CUDA

<https://developer.nvidia.com/cuda-gpus>，不过貌似在这里找不到也不一定代表不支持 CUDA，本人在显卡 MX150 上就有成功安装了 CUDA。

2.了解基础知识

1）**CUDA（Compute Unified Device Architecture）**，是显卡厂商 NVIDIA 推出的运算平台。 CUDA™是一种由NVIDIA 推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。

计算行业正在从只使用 CPU 的“中央处理”向 CPU 与 GPU 并用的“协同处理”发展。为打造这一全新的计算典范，NVIDIA™（英伟达™）发明了 CUDA（Compute Unified Device Architecture，统一计算设备架构）这一编程模型，是想在应用程序中充分利用 CPU 和 GPU 各自的优点。现在，该架构已应用于GeForce™（精视™）、ION™（翼扬™）、Quadro 以及 Tesla GPU（图形处理器）上。

2）**cuDNN** 的全称为 NVIDIA CUDA® Deep Neural Network library，是 NVIDIA 专门针对深度神经网络（Deep Neural Networks）中的基础操作而设计基于 GPU 的加速库。基本上所有的深度学习框架都支持 cuDNN 这一加速工具，例如：Caffe、Caffe2、TensorFlow、Torch、Pytorch、Theano 等。

3）Anaconda 是一个开源的 Python 发行版本，其包含了 conda、Python 等 180 多个科学包及其依赖项。因为包含了大量的科学包，Anaconda 的下载文件比较大，如果只需要某些包，或者需要节省带宽或存储空间，也可以使用 Miniconda 这个较小的发行版（仅包含 conda 和 Python）。

——from：<https://www.cnblogs.com/chamie/p/8707420.html>



# 三、Windows下安装tensorflow-gpu

## 1）需要下载

① NVIDIA 驱动程序下载地址：<https://www.nvidia.cn/Download/index.aspx?lang=cn>，进去会自动识别显卡型号

② CUDA 下载地址：<https://developer.nvidia.com/cuda-toolkit-archive>，如下（2019-06-21）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190621191552.png)

③ cuDNN 的下载地址：<https://developer.nvidia.com/rdp/cudnn-download>，如下（2019-06-21，需要注册账号才能下载）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190621192709.png)

点击 [Archived cuDNN Releases](https://developer.nvidia.com/rdp/cudnn-archive) 可以看到如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190621192413.png)



**注意1：** 担心在 windows 下安装软件出些幺蛾子，最好先安装好 **.Net Framework**。这是 .net framework 4.7.2 下载地址：[Download .NET Framework 4.7.2 | Free official downloads](https://dotnet.microsoft.com/download/dotnet-framework/net472)

**注意2：** GPU 显卡计算能力大于 3.0 才支持 cuDNN，查看 GPU 计算能力【https://developer.nvidia.com/cuda-gpus】

## 2）注意：版本问题

### 第一点：

注意：要知道自己电脑的 CUDA 版本号，则可以选择合适版本的 **CUDA Toolkit**，例如下图的 CUDA 版本号为 9.2（如何查看参考：[Windows系统查看CUDA版本号](https://www.jianshu.com/p/d3b9419a0f89)）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190621193427.png)

则可以安装 CUDA Toolkit 9.2、CUDA Toolkit 9.0、CUDA Toolkit 9.1、CUDA Toolkit 8.0 等（我的理解：即安装的版本不能超过截图看到的版本。）

### 第二点：

我们来看一篇文章的作者说的：

> **选择下载CUDA9.0**而不是CUDA10.0的原因：tensorflow_gpu库尚未发布与CUDA10.0对应的版本。本文作者写作此文时间是2018年11月14日，此时市面上tensorflow_gpu版本大多容易与CUDA9.0相配合。

说明也不能随便下载 cuda 的，需要根据你使用的 tensorflow-gpu 版本来决定下载哪个版本 cuda。

---

截止 2019-06-21 本人看到官方最新 tensorflow 版本为 2.0.0-alpha0。并且可以看到这里【[GPU 支持  |  TensorFlow](https://www.tensorflow.org/install/gpu?hl=zh-CN)】提到：

> TensorFlow 2.0 Alpha 可用于测试并支持 GPU。要进行安装，请运行以下命令：
>
> ``` python
> pip install tensorflow-gpu==2.0.0-alpha0
> ```

并且看到下面还写道，可以看到使用 tensorflow 2.0.0-alpha0 的要求：

> 必须在系统中安装以下 NVIDIA® 软件：
>
> - [NVIDIA® GPU 驱动程序](https://www.nvidia.com/drivers) - CUDA 10.0 需要 410.x 或更高版本。
> - [CUDA® 工具包](https://developer.nvidia.com/cuda-zone) - TensorFlow 支持 CUDA 10.0（TensorFlow 1.13.0 及更高版本）
> - CUDA 工具包附带的 [CUPTI](http://docs.nvidia.com/cuda/cupti/)。
> - [cuDNN SDK](https://developer.nvidia.com/cudnn)（7.4.1 及更高版本）
> - （可选）[TensorRT 5.0](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)，可缩短在某些模型上进行推断的延迟并提高吞吐量。

照文字意思我们可以理解为：

1. **安装 cuda 某个版本，那安装 NVIDIA 驱动版本也有要求**
2. t**ensorflow 2.0.0-alpha0，支持 cuda 10.0，cuDNN 7.4.1 及以上。**
3. **tensorflow 1.13.0 及以上，都支持 cuda 10.0**

补充，查看已经安装的驱动的版本方法，在【设备管理器】找到要查看的驱动，右键驱动【属性】，切换到【驱动程序】，可以看到如下截图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190621205233.png)



### 我个人的理解和小结（仅参考）

1、先打开你电脑的【NVIDA 控制面板】->【系统信息】->【组件】查看 `NVCUDA.DLL` 这项的产品名称，其中名称中的版本表示你安装的 CUDA 不能超过它。

2、然后根据你要安装的 tensorflow 版本所支持的 CUDA 版本。比如 `tensorflow-gpu 2.0.0-alpha0` 是支持 CUDA 10.0 的，那么你可以下载 CUDA 10.0，这里也要提下：安装 CUDA 也要和 GPU 驱动版本对应。可以见下文我的“第二次安装记录：win10+MX150下安装tensorflow-gpu”有遇到的问题。

3、然后根据你要安装的 tensorflow 查看所支持的 cuDNN 版本，比如 `tensorflow-gpu 2.0.0-alpha0` 支持 cuDNN 7.4.1 及更高版本

4、选择出 cuDNN 版本后，然后进【<https://developer.nvidia.com/rdp/cudnn-download>】选择 cuDNN for CUDA 版本。

## 3）最后：命令安装 tensorflow

前面各个软件和工作的我就在此安装省略了。

windows 下安装：

- CPU版：`pip install --upgrade tensorflow`
- GPU版：`pip install --upgrade tensorflow-gpu`



## 4）第一次安装记录：win7+GTX 1080ti下安装tensorflow-gpu

电脑配置和已有环境：

- Windows7-64bit

- GTX 1080ti 显卡

- 已安装 `Anconda3-5.1.0-Windows-x86_64.exe`

  

### 1 安装CUDA 10.0

首选我根据电脑显卡选择了 CUDA Toolkit 10.0： `cuda_10.0.130_411.31_windows.exe` 安装，但安装过程未成功，显示很多组件未安装，如下图（来源于网上）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190622134432.png)

后来看了参考文章[4]中提到的这篇文章：<https://blog.csdn.net/leelitian3/article/details/83272272>，这里提到必须安装 **Visual Studio**，CUDA 需要与其结合使用。

> ①Visual Studio 2017 Community（必须安装，Cuda是与其结合使用的）
>
> - 下载地址：<https://visualstudio.microsoft.com/zh-hans/free-developer-offers/>
> - 安装选项：勾选“C++的桌面开发”即可

于是我照做了，下载了 Visual Studio，勾选了“C++的桌面开发”安装，最后解决了 CUDA 安装失败的问题。

这里我提下，我安装的 VS 版本是： Visual Studio 2019 Community，采用的是离线安装方式，如何离线安装请参考：

- [Visual Studio 2017离线安装包获取和安装教程](https://blog.csdn.net/WU9797/article/details/78456651)
- [Microsoft Visual Studio 2019正式版离线安装包下载](https://www.bitecho.net/microsoft-visual-studio-2019.html)

大概就是：

1. 先去官网 <https://visualstudio.microsoft.com/zh-hans/downloads/> 下载所需版本的安装包获取程序，比如我下载 VS2019 社区版安装包获取程序：`vs_community__1035315853.1561183168.exe`

2. 然后通过这个执行这个「获取程序」去下载 Visual Studio 安装包，命令参数有很多，如：

   ``` xml
   vs_community__1035315853.1561183168.exe --layout D:\VS2017社区版 --lang zh-CN 
   ```

   表示下载到到目录是：G:\VS2017专业版Offline；表示软件的语言包是：中文

3. 然后下载完成之后，进入安装包目录，点击相应 exe 文件运行，后面会出来让你选择哪些 Visual Studio 组件安装。对于如上我安装 CUDA 失败的问题，我选择了“使用 C++ 的桌面开发”。

注：对于如上 CUDA 安装过程组件未安装的问题，我看到这篇文章也提到：[windows安装CUDA 10自定义安装出现错误组件未安装解决方案](https://blog.csdn.net/weixin_44146276/article/details/86703067)，它的解决方法是：

> 我们只用选择 CUDA下面这 4 项就够了（默认时是全选的），**visual studio integration这一项别勾选**是因为可能我们电脑并没有使用 VS 环境，如下图所示。
>
> ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190622141705.png)

上面该解决方法做参考用。

安装完 CUDA，最后，打开命令提示符，输入**nvcc -V**，出现如下类似信息即为成功：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190622142002.png)

##### 安装CUDA这里提下这篇文章，做参考

[TensorFlow2.0 系列开篇： Windows下GPU版本详细安装教程 - 知乎](<https://zhuanlan.zhihu.com/p/71030147>)

上面这篇文章有写道如何安装 CUDA：

官网链接：<https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal>，下载完成后，打开下载的驱动，取勾 GeForce Experience

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190704102016.png)

如果电脑上本身就有 Visual Studio Integration，要将这个取消勾选，避免冲突了：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190704102034.png)

点开 Driver comonents，Display Driver 这一行，前面显示的是 CUDA 本身包含的驱动版本是 `411.31`

如果你电脑目前安装的驱动版本号新于 CUDA 本身自带的驱动版本号，那一定要把这个勾去掉。否则会安装失败(相同的话，就不用去取勾了)

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190704102113.png)

接下来需要几分钟的时间安装。

------------------------分割线------------------------------

**从如上这篇文章我们似乎可以看到，安装 CUDA 好像其实并不需要安装 Visual Studio。另外，下载的 CUDA 默认是有显卡驱动的。可以不用去单独下载显卡驱动安装。**

------------------------分割线------------------------------

安装完 CUDA，那基本快完成了。接下来安装好 cuDNN 即可，其实也就是拷贝几个文件到 CUDA 安装目录对应的文件夹下即可。请看下面内容。最后命令安装 tensorflow-gpu，完成！



### 2 安装 cuDNN：cuDNN v7.6.0 for CUDA 10.0

解压出来下载的 `cudnn-10.0-windows7-x64-v7.6.0.64.zip`，根据参考文章 [4] 中，把相应的文件拷贝到合适的目录即可。

这里最简便的方法是，解压出来 cuDNN 的 zip 文件，会看到有 3 个文件夹：**bin、include、lib**，和 1 个文件**NVIDIA_SLA_cuDNN_Support.txt**，选中这 4 个文件夹和文件，然后复制，找到 CUDA 安装目录，粘贴到对应的 CUDA 目录，即可完成 cuDNN 的安装。



### 3 命令安装 tensorflow-gpu

进入你需要安装 tensorflow-gpu 到的那个环境，使用:

``` python
pip install --upgrade tensorflow-gpu
```

即可安装 tensorflow GPU 版。

> 注：本人安装时间为 2019-06-22，使用上面命令安装 tensorflow，默认给我安装的版本为 1.14.0

你也可以指定想要安装的版本，如 `pip install tensorflow-gpu==1.2.1`， == 后面为所要安装的版本号。

后面，本人有 conda 命令再新建了一个名为 tf2 的环境：`conda create -n tf2 python=3.6.4`，然后进入该环境，使用 pip 安装 tensorflow2.0：

``` python
pip install tensorflow-gpu==2.0.0-alpha0
```

成功安装好了 tensorflow2.0。测试：

```
import tensorflow as tf
print(tf.__version__)
```

结果如下：

``` xml
2.0.0-alpha0
```

即成功安装 tensorflow2.0 到了 tf2 环境下。然后你可以在该环境下安装你需要的库等。

这里提一个小技巧：对于安装各种 Python 库，每次都需要在 cmd 下，然后使用 activate 进入，我是真闲麻烦，这里我有个小技巧，你可以新建一个 `.bat` 文件，放在桌面，然后双击可以直接进入你的环境。bat 文件内容如下：

``` bash
@echo off
cmd /k activate keras
```

## 5）第二次安装记录：win10+MX150下安装tensorflow-gpu

本人有在显卡为 MX150 的笔记本尝试安装 tensorflow-gpu，下载了 cuda 9.0 进行安装，发现安装过程并没有出现组件未安装等问题，都是顺利进行，未安装 Visual Studio，直接是安装完 cuda 9.0 然后安装 cuDNN，然后 `pip install tensorflow-gpu`，顺利完成了 tensorflow-gpu 安装过程。使用过程中执行：

```python
import tensorflow as tf
print(tf.__version__)
```

是没有报任何错误的，按理是表明安装成功了。但是后面运行如下代码：

```python
import tensorflow as tf

a = [1.0, 2.0, 3.0, 4.0]
with tf.Session() as sess:
    print(sess.run(tf.nn.softmax(a)))
```

会报如下错误：

```xml
CUDA driver version is insufficient for CUDA runtime version
```

翻译过来意思是：CUDA驱动程序版本不足以用于CUDA运行时版本。

遂，我查看了下我的笔记本显卡驱动版本，为如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190703205225.png)

可以看到驱动版本为 `22.21.13.8554`。这里提一下查看显卡驱动的方法，有两种：

> 1. Win+ R 组合键 -> 键入 `msinfo32` -> 按 Enter 键 -> 导航至"组件部分，并选择"显示"：列出的"驱动程序版本即为驱动程序版本。
> 2. 右键“计算机” -> "属性" -> "设备管理器" -> "显示适配器" -> 选择对应的显卡驱动，右键 -> “属性” -> “驱动版本”，然后就可以看到驱动程序版本是多少。

本人尝试去更新驱动版本，按照如上第二个方法，我在“驱动版本”界面，选择了“更新驱动程序(P)”，然后选择“自动搜索更新的驱动程序软件(S)”进行驱动更新。最后我的驱动版本为 `25.21.14.1972`，如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190703210313.png)

驱动更新完，再去执行前面那段程序，运行成功。

**如上问题，正说明 CUDA 版本对 GPU 的驱动版本也是有要求的。**这个时候可以再回头看下本人记录的“版本问题”第二点下面记录的。

可参考下：[解决CUDA driver version is insufficient for CUDA runtime version](https://www.cnblogs.com/liaohuiqiang/p/9791365.html)  |  [CUDA、显卡驱动和Tensorflow版本之间的对应关系](<https://blog.csdn.net/IT_xiao_bai/article/details/88342921>)

> 可以看到上面博文里的截图的显卡驱动版本都是 300 多的数值，我还没明白这个的版本是看的哪个值，我在 windows 下看显卡驱版本没看到这样的版本数值呢。
>
> update：2019-07-04 我现在知道了，上面博文里显卡的驱动版本的数值其实就是上面看到的驱动版本对应的后五位数。如 `22.21.13.8554`，对应的版本是 `385.54`，这个版本值在 `NVIDIA` 控制面板可以看到。





## 6）参考文章

- [1] [win7 64位+CUDA 9.0+cuDNN v7.0.5 安装](https://blog.csdn.net/shanglianlm/article/details/79404703)  [荐] 
- [2] [这是一份你们需要的Windows版深度学习软件安装指南](https://zhuanlan.zhihu.com/p/29903472)  [荐]
- [3] [深度学习环境搭建-CUDA9.0、cudnn7.3、tensorflow_gpu1.10的安装](<https://www.jianshu.com/p/4ebaa78e0233>)  [荐]
- [4] [安装最新版tensorflow Cuda10.0 cuDNN Win10 VS 2017 - 知乎](https://zhuanlan.zhihu.com/p/49832216)  [荐]



# 四、Ubuntu下安装tensorflow-gpu


参考：

- [从零开始搭建深度学习服务器: 基础环境配置（Ubuntu + GTX 1080 TI + CUDA + cuDNN）](http://www.52nlp.cn/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%90%AD%E5%BB%BA%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%9C%8D%E5%8A%A1%E5%99%A8%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AEubuntu-1080ti-cuda-cudnn)

