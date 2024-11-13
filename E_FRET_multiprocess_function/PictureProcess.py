import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg


class PictureProcess:
    """
    图像处理工具类
    """

    @staticmethod
    def read_image(curr_subdir_path: str):
        """
        读取子文件夹中的三通道图片
        :param curr_subdir_path: 子文件路径
        :return: 三通道图像数组
        """
        dd = mpimg.imread(os.path.join(curr_subdir_path, 'DD.tif'))
        da = mpimg.imread(os.path.join(curr_subdir_path, 'DA.tif'))
        aa = mpimg.imread(os.path.join(curr_subdir_path, 'AA.tif'))
        return dd, da, aa

    @staticmethod
    def save_image(curr_subdir_path, matrix, filename, filetype: str = "gray"):
        """
        图像保存函数
        :param curr_subdir_path: 子路径文件夹
        :param matrix: 输入的图像矩阵
        :param filename: 文件名称
        :param filetype: 文件类型 一般是灰度图
        """
        plt.imsave(os.path.join(curr_subdir_path, filename), matrix, cmap=filetype)

    @staticmethod
    def figure_gray_pic_and_pcolor_pic(matrix, result_dir_path, name):
        """
        绘制分布灰度图和伪彩图
        :param matrix: 矩阵
        :param result_dir_path: 保存路径
        :param name: 对象名字
        """
        plt.imsave(os.path.join(result_dir_path, name + ".jpg"), matrix, cmap='gray')
        imgGcf = plt.figure()
        plt.pcolor(np.flipud(matrix))
        plt.axis('off')
        plt.title(name + '伪彩图像', fontproperties='SimHei')
        plt.colorbar()
        plt.close()
        imgGcf.savefig(os.path.join(result_dir_path, name + "伪彩图.jpg"))

    @staticmethod
    def figure_hist(matrix, result_dir_path, name, start, stop, step, threshold):
        """
        绘制统计图
        :param step:
        :param stop:
        :param start:
        :param name: 图像名称
        :param matrix: 矩阵
        :param result_dir_path: 存储路径
        :return: 去除极大极小值的矩阵
        """
        ae = matrix[matrix > 0]
        ae = ae[ae < threshold]
        bE = np.arange(start, stop, step)
        m1, n1, pathches = plt.hist(ae, bE, color='black')
        maxAVG = max(m1)
        if maxAVG == 0:
            print("不存在具有效率或者浓度的点")
            return ae

        imgGcf = plt.figure()
        m2 = m1 / maxAVG
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.ylabel('frequency', size=20)
        plt.xlabel(name, size=15)
        plt.title(name + "直方图", fontproperties='SimHei')
        n2 = np.arange(start, stop - step, step)
        plt.bar(n2, m2, width=step - step / 0.05 * 0.01, color='black')
        plt.close()
        hist_dir = os.path.join(result_dir_path, name + '_hist.jpg')
        imgGcf.savefig(hist_dir)
        return ae
