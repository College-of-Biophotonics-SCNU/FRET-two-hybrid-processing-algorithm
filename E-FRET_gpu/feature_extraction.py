"""
基于 Ed 效率图像进行特征提取操作
1. 获取聚点位置，采用梯度下降算法获取对应的聚点区域范围
2. 统计聚点的平均效率情况
"""

import torch
import torchvision
import numpy as np
import cv2



