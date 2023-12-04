from datetime import datetime

import numpy as np


def isEqual(num1, num2):
    """
    比较判断
    :param num1: 比较参数1
    :param num2: 比较参数2
    :return: 比较结果
    """
    if num1 == num2:
        return True
    return False


def calculate_mean_and_std(matrix):
    """
    计算均值和标准差
    :param matrix: 矩阵
    :return: 均值，标准差
    """
    mean = np.mean(matrix)
    std = np.std(matrix, ddof=1)
    return mean, std


def cost_time(start_time):
    return (datetime.now() - start_time).seconds
