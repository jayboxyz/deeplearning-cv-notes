

## conda常用命令速查



``` 
conda -V 或 conda --version	#查看版本
conda --help  / conda -h  如：conda env -h

conda update conda	#更新conda

conda create --name <env_name> <package_names> #创建环境，如 conda create --n mykeras python=3.6.4
conda remove --name <env_name> --all #删除环境

conda info --envs 或 conda env list #显示已创建环境
activate env-name #Linux/Mac: source activate <env_name> # 进入环境
deactivate	#Linux/Mac: source deactivate 退出

conda create --name flowers --clone snowflakes	#制作环境的完整副本。这里我们将克隆snowflakes创建一个名为flowers的精确副本：

#更换源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes

conda search numpy	# 查找package信息
conda search --full-name python	#查找全名为“python”的包有哪些版本可供安装。

conda install -n py3 numpy	# 安装package到名为py3环境,如果不用-n指定环境名称，则被安装在当前活跃环境
conda install scipy	# 安装scipy到当前活跃环境
conda update --all 或者 conda upgrade --all	# 更新所有包
conda update -n py3 numpy	# 更新名为py3环境的package
conda remove -n py3 numpy	# 删除名为py3环境的package

##conda 将 conda、python 等都视为 package：
conda update conda # 更新conda，保持conda最新
conda update anaconda # 更新anaconda
conda update python # 更新python

分享环境：
conda env export > environment.yml
小伙伴拿到environment.yml文件后，将该文件放在工作目录下，可以通过以下命令从该文件创建环境：
conda env create -f environment.yml


conda list	#列举当前活跃环境下的所有包
conda list -n your_env_name		#列举一个非当前活跃环境下的所有包
conda install -n env_name package_name	#为指定环境安装某个包
```



## pip 常用命令速查



``` 
pip install <package>	#


```

