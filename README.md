# 一、简介
本项目是移动通信网络流量预测的示例代码，请同学们结合课堂学习知识，跟着readme进行操作，仔细阅读代码注释，尝试更改参数，观察实验结果，总结分析，完成课程任务。
# 二、项目结构
**Data文件夹**；包含了聚合后的数据集、预处理数据、模型预测的结果数据、标签数据

**Model文件夹**： 保存了深度学习模型训练后的权重参数文件

**results文件夹**：保存了所有代码在运行过程中输出的结果图篇

**0_Data_Visualization.py**：用于观察数据的代码实现

**1_HA.py**：基于历史平均算法进行流量预测的代码实现

**2_ARIMA.py**：基于差分整合移动平均自回归算法进行流量预测的代码实现

**3_LSTM.py**：基于长短期记忆网络搭建深度模型进行流量预测的代码实现

**4_DeepST.py**：基于卷积神经网络搭建深度模型（MiniDeepst、MiniSTResNet、MiniSTResNet+Dropout）进行流量预测的代码实现

**5_ConvLSTM**：基于卷积长短期记忆网络搭建模型（ConvLSTM、ConvLSTM+Dropout）进行流量预测的代码实现

**BuildModel.py**：自定义模型函数库

**DataPrePrescess**：自定义数据预处理函数库

**requirements.txt**：给出了运行该项目代码所需要的安装的python库

**SimHei.ttf**：字体文件，用于画图时显示中文字体

**TrainModel.py**：自定义模型训练时的函数库

**Visualization**：自定义用于可视化的函数库

# 三、运行准备
需要安装pycharm和Anaconda，参考教程：http://t.csdnimg.cn/OjXH3
若想运行本项目的代码，你需要：\
使用pycharm打开工程文件，
打开终端，使用conda命令创建虚拟环境
```
conda create -n 虚拟环境名称 python=3.8 
```
激活虚拟环境
```
conda activate 虚拟环境名称
```
使用pip命令安装所需库
```
pip install -r requirements.txt
```
# 四、动手实践
## 基础篇
### 0、观察数据
在项目根目录下执行下列指令
```
python 0_Data_Visualization.py
```
### 1、使用HA算法进行流量预测
在项目根目录下执行下列指令
```
python 1_HA.py
```
### 2、使用ARIMA算法进行流量预测
在项目根目录下执行下列指令
```
python 2_ARIMA.py
```
## 进阶篇
### 3、使用LSTM模型进行流量预测
在项目根目录下执行下列指令
```
python 3_LSTM.py
```
### 4、使用DeepST模型进行流量预测
在项目根目录下执行下列指令
```
python 4_DeepST.py
```
## 探索篇
### 5、使用ConvLSTM模型进行流量预测
在项目根目录下执行下列指令\
```
python 5_ConvLSTM.py
```
