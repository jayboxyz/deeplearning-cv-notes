matlab 2017a 安装教程：

- 下载和破解：[Matlab R2017a 中文版下载安装与破解图文教程](<https://blog.csdn.net/gisboygogogo/article/details/76793803>)
- 按照上面的步骤安装后提示”License Manager Error – 8″，则按这个方式解决：[MathWorks MATLAB R2017a 官方原版+完美破解补丁 | 乐软博客](<https://www.isharepc.com/2073.html>)

# matlab 常见的图像处理操作

1、imread() 函数

image = imread(filename) 可以读取图像并查看返回的数值，如果想要查看指定维度的图像数值，可以：

``` python
image = imread(filename)(:, :, 3)  #返回第三维度的数值
```





# 一、matlab 基础

## 1. matlab 认识

（1）M 文件

> 用 MATLAB 语言编写的程序，称为 M 文件。M 文件可以根据调用方式的不同分为两类：
>
> - **命令(脚本)文件**(Script File)和
> - **函数文件**(Function File)。
>
> M 文件是由一系列 MATLAB 语句组成的文件，包括命令文件和函数文件两类，命令文件类似于其他高级语言中的主程序或主函数，而函数文件则类似于子程序或被调函数。

（2）脚本文件

> 当调用脚本时，MATLAB 仅执行在文件中找到的命令。脚本可以处理工作区中的现有数据，也可以创建要在其中运行脚本的新数据。尽管脚本不会返回输出参数，其创建的任何变量都会保留在工作区中，以便在后续计算中使用。此外，脚本可以使用 plot 等函数生成图形输出。

（3）函数文件

> 函数是可接受输入参数并返回输出参数的文件。文件名和函数名称应当相同。函数处理其自己的工作区中的变量，此工作区不同于您在 MATLAB 命令提示符下访问的工作区。

示例：(rank.m)
``` matlab
function r = rank(A,tol)

s = svd(A);
if nargin==1
   tol = max(size(A)') * max(s) * eps;
end
r = sum(s > tol);
```

函数的第一行以关键字 function 开头。它提供函数名称和参数顺序。本示例中具有两个输入参数和一个输出参数。

文件的其余部分是用于定义函数的可执行 MATLAB 代码。函数体中引入的变量 s 以及第一行中的变量（即 r、A 和 tol）均为函数的局部变量；他们不同于 MATLAB 工作区中的任何变量。

本示例演示了 MATLAB 函数不同于其他编程语言函数的一个方面，即可变数目的参数。可以采用多种不同方法使用 rank 函数：
``` matlab
rank(A)
r = rank(A)
r = rank(A,1.e-6)
```
许多函数都按此方式运行。如果未提供输出参数，结果会存储在 ans 中。如果未提供第二个输入参数，此函数会运用默认值进行计算。函数体中提供了两个名为 nargin 和 nargout 的数量，用于告知与函数的每次特定使用相关的输入和输出参数的数目。rank 函数使用 nargin，而不需要使用 nargout。

（4）匿名函数：`f = @(arglist)expression`

下面的语句创建一个求某个数字的平方的匿名函数。当调用此函数时，MATLAB 会将您传入的值赋值给变量 x，然后在方程 x.^2 中使用 x：`sqr = @(x) x.^2;`

（5）空参函数

``` xml
function [] = funcname() 
    % function funcname
...
...
end
```

**注：**在一个 m 文件中，可以定义多个函数，但是文件名一定要与第一个函数（主函数）一致，该文件中其他函数则为本文件的私有函数，外部不可访问(可以通过参数调用的方法访问)。

（5）元胞数组

> 元胞数组是 MATLAB 的一种特殊数据类型，可以将元胞数组看做一种无所不包的通用矩阵，或者叫做广义矩阵。组成元胞数组的元素可以是任何一种数据类型的常数或者常量，每一个元素也可以具有不同的尺寸和内存占用空间，每一个元素的内容也可以完全不同，所以元胞数组的元素叫做元胞（cell）。和一般的数值矩阵一样，元胞数组的内存空间也是动态分配的。
>
> 创建：` a={'matlab',20; ones(2,3),1:10}`



## 2. matlab 快捷键

代码自动补全：按 `tab` 键。

代码注释方法1：
``` xml
%{

若干语句

%}
```
代码注释方法2：
- 多行注释: 选中要注释的若干语句, 编辑器菜单Text->Comment, 或者快捷键Ctrl+R

- 取消注释: 选中要取消注释的语句, 编辑器菜单Text->Uncomment, 或者快捷键Ctrl+T

代码注释方法3：
``` xml
if LOGICAL(0)

若干语句

end
```
这个方法实际上是通过逻辑判断语句不执行相关的语句

## 3. matlab 调试

点击在要设置断点的行左侧的-，将其变成圆圈后即设置了断点，也可按 F12 设置断点。条件断点在 debug 菜单下的 set/modify conditional breakpoint 进行设置。

设置断点后按 F5 运行程序，会在断点处停止运行，按 F10 可以单步运行调试，以上内容也可在 Debug 菜单下和工具条（第一条）中找到。

运行速度我不知道你指的是什么，我给出一个。当按下两个%后，即定义了一个 Cell 块，定义多个块后，当改变了某处的程序，可以有选择的从指定块后运行而不必重头开始运行程序，提高程序运行效率。以上内容可在 Cell 菜单下和工具条（第二条）中找到。



# 二、matlab 常用命令

经常在一些 matlab 程序中看到如下：

``` xml
clc：清除命令窗口的内容，对工作环境中的全部变量无任何影响 
close：关闭当前的Figure窗口 
close all:关闭所有的Figure窗口 
clear：清除工作空间的所有变量 
clear all：清除工作空间的所有变量，函数，和MEX文件
```

- 在命令窗口中输入path，就能查看 matlab 的搜索路径（比如，输入一条代码，程序会在这些目录中依次搜索是否为变量，是否为函数，M文件等）。

- `cd`：查看当前工作目录
- `userpath('F:\matlab\work')`:修改工作路径
- `savepath`：保存路径修改

    > 改完默认路径后需要保存一下。
- `addpath`：`addpath('D:\Workspace\Matlab\DL\DeepLearnToolbox-master');`

    > 该路径添加到搜索路径 后:`addpath('D:\Workspace\Matlab\DL\DeepLearnToolbox-master','-end');`

- `who`:显示所有使用过的变量名
- `whos`:更多的显示变量
- `...`:长任务，一行写不下，多行写
- `clear`:清除变量（从内存中删除变量）
- `format long`:命令显示十进制后的16位数字。(matla默认显示四位小数)
- `clc`:清除命令窗口
- `exist`:检查文件或变量是否存在
- 创建一个名为progs的文件夹：
    ``` html
    mkdir progs   %创建文件夹
    chdir progs    %进入新创建的文件夹
    edit  prog1.m  %创建并编辑
    prog1  %运行
    ```

- `format short`:matlab默认显示四位小数位，这称为短格式
- `format long`:显示十进制后的16位数字
- `format bank`:将数字舍入到小数点后两位
- `format short e`:命令以指数形式显示四位小数加上指数。
- `format rat`:命令给出计算结果最接近的合理表达式
    ``` xml
    Trial>> format rat
    4.678 * 4.9
    
    ans =
    
        2063/90
    ```

- `type myfunction`:要查看程序文件（例如，myfunction.m）的内容

最常用 matlab 基本数据类型：https://www.yiibai.com/matlab/matlab_data_types.html#article-start



# 三、matlab 常用函数

- `uigetfile`：[uigetfile](https://ww2.mathworks.cn/help/matlab/ref/uigetfile.html)

    > 打开文件选择对话框。

- `imadjust`：[imadjust](http://blog.sina.com.cn/s/blog_14d1511ee0102ww6s.html)

    > `f1=imadjust（f，[low_in  high_in],[low_out  high_out],gamma）` 该函数的意义如图1所示，把图像f 灰度变换到新图像f1的过程中，f 中灰度值低于low_in的像素点在f1中灰度值被赋值为low_out,同理，f中灰度值高于high_in的像素点变换到f1时其灰度值也被赋值为high_out;而对于参数gamma，当gamma<1时，灰度图像靠近low_in的灰度值较低像素点灰度值变高，其灰度变化范围被拉伸，灰度值靠近high_in的一端灰度变化范围被压缩，图像整体变明亮。

- `imread`：

    > imread函数用于读入各种图像文件，其一般的用法为`[X，MAP]=imread(‘filename’,‘fmt’)`其中，X，MAP分别为读出的图像数据和颜色表数据，fmt为图像的格式，filename为读取的图像文件（可以加上文件的路径）。

- `imwrite`：

    > imwrite函数用于输出图像，其语法格式为：`imwrite(X,map,filename,fmt)`按照fmt指定的格式将图像数据矩阵X和调色板map写入文件filename。

- `[M,N]=size(img)`：

    > 

- `figure`：

    > 

- `subplot`：

    > 

- `imagesc`：

    > 

- `rgb2gray`：

    > 

- `[M,N]=size(img)`：

    > 

绘图：
- `plot`：[plot](https://ww2.mathworks.cn/help/matlab/ref/plot.html?searchHighlight=plot&s_tid=doc_srchtitle)

    > 二维线图。plot(X,Y) 创建 Y 中数据对 X 中对应值的二维线图。
- [Matlab中plot、fplot、ezplot的使用方法和区别](https://www.cnblogs.com/lihuidashen/p/3443337.html)

- `grid on`、`grid off`：

    > 显示或隐藏坐标区网格线。grid on 显示 gca 命令返回的当前坐标区或图的主网格线。主网格线从每个刻度线延伸。

- `hold on`、`hold off`、：添加新绘图时保留当前绘图。[hold](https://ww2.mathworks.cn/help/matlab/ref/hold.html?searchHighlight=hold%20on&s_tid=doc_srchtitle)

- `zeros`：创建全零数组

- `axes`：创建笛卡尔坐标区

- `fopen`：打开文件，示例 `fid = fopen(filename,'wt');`

- `strsplit`：在指定的分隔符处拆分字符串，示例 `newname = strsplit(filename,'.')`
- `fullfile`：从各个部分构建完整文件名

    > `f = fullfile(filepart1,...,filepartN)` 根据指定的文件夹和文件名构建完整的文件设定。fullfile 在必要情况下插入依平台而定的文件分隔符，但不添加尾随的文件分隔符。在 Windows 平台上，文件分隔符为反斜杠 (\\)。在其他平台上，文件分隔符可能为不同字符。

一些技巧：

- 字符串拼接：比如 filename='C:\' pathname='a.jpg'

  - path = [filename, pathname]
  - path = strcat(pathname,filename)

- 各种类型之间的转换：`num2str`将数字转换为字符数组、`str2num`将数字转换为字符数组
- `deblank`：删除字符串或字符数组末尾的尾随空白



# 四、matlab GUI 编程

**回调函数：**

每个控件都有几种回调函数，右键选中的控件一般会有如下菜单：

- CreateFcn 是在控件对象创建的时候发生(一般为初始化样式，颜色，初始值等)
- DeleteFcn 实在空间对象被清除的时候发生
- ButtonDownFcn和KeyPressFcn分别为鼠标点击和按键事件Callback
- CallBack为一般回调函数，因不同的控件而已异。例如按钮被按下时发生，下拉框改变值时发生，sliderbar 拖动时发生等等。

然后就可以跳转到相应的 Editor中编辑代码，GUIDE会自动生成 相应的函数体，函数名，名称一般是控件 Tag+ Call类型名 参数有三个 ( hObject, eventdata, handles)
>其中 hObject 为发生事件的源控件，eventdata为事件数据结构，handles为传入的对象句柄 
>
>[MATLAB GUI handles与hObject的区别理解](https://blog.csdn.net/qq_20823641/article/details/51865384)


**Matlab关于Figure 和Axes的区别：**

- [Matlab关于Figure 和Axes的区别](https://zhidao.baidu.com/question/1755065950881292188.html?qbl=relate_question_7)
- [正确解释：关于Figure 和Axes的区别](http://www.ilovematlab.cn/thread-52140-1-1.html)
- [MATLAB中axes函数全功能解析](http://blog.sina.com.cn/s/blog_61c0518f0100f5rb.html)

gca 函数：

    ax = gca 返回当前图窗的当前坐标区或图，这通常是最后创建的图窗或用鼠标点击的最后一个图窗。图形函数（例如 title）的目标为当前坐标区或图。可以使用 ax 访问和修改该坐标区或图的属性。如果该坐标区或图不存在，gca 将创建笛卡尔坐标区。


**Matlab GUI 程序例子：**

- [Matlab GUI入门获取\设置界面控件的值](https://blog.csdn.net/ghevinn/article/details/45973679)：为matlab GUI 计算器简单例子。
- [MATLAB GUI uitable 如何处理显示excel数据？](https://jingyan.baidu.com/article/a17d5285e697d58099c8f271.html)

**Matlab的GUI参数传递：**

- [Matlab的GUI参数传递方式总结](https://blog.csdn.net/SMF0504/article/details/51814375)


**GUI弹出框提示：**

- [matlab GUI之常用对话框（三）--- dialog \ errordlg \ warndlg \ helpdlg \ msgbox \questdlg](https://blog.csdn.net/zjq2010014137/article/details/8535431)

1. 普通对话框 dialog 
    ``` matlab
    %普通对话框  
    h=dialog('name','关于...','position',[200 200 200 70]);  
      
    uicontrol('parent',h,'style','text','string','你好！','position',[50 40 120 20],'fontsize',12);  
    uicontrol('parent',h,'style','pushbutton','position',...  
       [80 10 50 20],'string','确定','callback','delete(gcbf)');  
    ```

2. 错误对话框
    ``` matlab
    %错误对话框  
    h=errordlg('警告','错误');  
    ha=get(h,'children');  
      
    hu=findall(allchild(h),'style','pushbutton');  
    set(hu,'string','确定');  
    ht=findall(ha,'type','text');  
    set(ht,'fontsize',20,'fontname','隶书'); 
    ```

3. 警告对话框
    ``` matlab
    %警告对话框  
    h=warndlg('内存不足','警告','modal'); 
    ```

4. 帮助对话框
    ``` matlab
    %帮助对话框  
    helpdlg('双击对象进入编辑状态','提示');  
    ```
5. 信息对话框
    ``` matlab
    %信息对话框  
    msgbox('中日钓鱼岛之争愈演愈烈！','每日新闻','warn');  
    ```
    Icon中有‘error’、‘warn’、‘help’、‘custom’
6. 提问对话框
    ``` matlab
    %提问对话框  
    questdlg('今天你学习了吗？','问题提示','Yes','No','Yes');  
    ```