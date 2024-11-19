# 项目名称

一个使用机器学习预测奥运会项目的工具

## 项目简介

本项目使用熵权法、TOPSIS 方法以及最小二乘法构建一个多层次模型，用于预测奥运会是否会包含某个项目。输入是某项目的若干个小指标，通过滑动窗口的方法使用前两年预测后一年的数据，并结合各小指标的权重进行加权，最终通过模型输出是否会包含该项目。

## 安装与依赖

### 环境要求

- Python 3.x
- 需要安装以下依赖库：

```bash
pip install -r requirements.txt
