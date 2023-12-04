import multiprocessing
import os
import time
from datetime import datetime

import numpy as np
from FileProcess import *
from tqdm import tqdm
from BaseComputer import *
from PictureProcess import PictureProcess


class FRET:
    """
    E-FRET 计算
    """
    # 类变量，用来记录进度条进程
    pbar = {}

    def __init__(self,
                 a: int = 0.177327,
                 b: int = 0.00997,
                 c: int = 0.001244,
                 d: int = 0.791588,
                 G: int = 7.735335,
                 k: int = 0.538395,
                 expose_times: tuple = (300, 300, 300),
                 BACKGROUND_THRESHOLD: int = 1.5,
                 pcolor: bool = False,
                 main_dir: str = ''
                 ):
        """
        :param a: 校正因子a
        :param b: 校正因子b
        :param c: 校正因子c
        :param d: 校正因子d
        :param G: 校正因子G
        :param k: 校正因子k
        :param BACKGROUND_THRESHOLD: 背景模板波长阈值超参数
        :param expose_times: 三通道曝光时间
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.G = G
        self.k = k
        self.expose_times = expose_times
        self.BACKGROUND_THRESHOLD = BACKGROUND_THRESHOLD
        self.pcolor = pcolor
        self.subdir_list = []  # 子目录列表
        self.subdir_len = 0  # 子目录长度
        self.main_dir = main_dir

    @staticmethod
    def initialize_background(background_matrix):
        """
        初始化背景矩阵
        :param background_matrix: 背景矩阵维度
        :return: 背景模板
        """
        shape = (background_matrix[0], background_matrix[1], 3)
        background_base_mould = np.ones(shape, dtype=np.double)
        return background_base_mould

    @staticmethod
    def build_image_matrix(dd, da, aa):
        """
        构建三通道图像矩阵
        :param m: 矩阵维度
        :param dd: DD通道图像
        :param da: DA通道图像
        :param aa: AA通道图像
        :return: 三通道图像矩阵
        """
        image = np.ones((dd.shape[0], dd.shape[1], 3), dtype=np.double)
        image[:, :, 0] = dd
        image[:, :, 1] = da
        image[:, :, 2] = aa
        return image

    def calculate_background(self, original_channels, background_base, result_dir):
        """
        分别确定三个通道的背景
        :param result_dir: 背景模板保存路径
        :param original_channels: 初始三通道强度矩阵
        :param background_base: 背景基底矩阵
        :return: 去背景后强度矩阵，三通道背景模板矩阵
        """
        # 参数定义为去除背景的三通道
        no_background_channels = np.zeros(original_channels.shape, dtype=np.double)
        for i in range(3):
            one_channel = original_channels[:, :, i]
            # 统计函数，统计0~2000中存在背景图像最可能的值
            one_channel = one_channel[one_channel > 0]
            one_channel = one_channel[one_channel < 2000]
            bg_hist, bg_bin_edges = np.histogram(one_channel, bins=np.arange(1, 2000, 1))
            # 这里考虑的情况是，背景值的像素一定是最多的
            max_count = max(bg_hist)
            pixel_list = np.where(bg_hist == max_count)
            # 将图片扣去背景
            background_base[:, :, i] = (original_channels[:, :, i] - background_base[:, :, i]
                                        * np.max(pixel_list) * self.BACKGROUND_THRESHOLD)
            background_base[background_base < 0] = 0
            background_base[background_base > 50000] = 0

        # 对通道曝光时间进行归一化操作, 这里图像是扣去背景的
        no_background_channels = np.divide(no_background_channels, self.expose_times)
        # 模板合并操作 对于背景模板图像来说，这里应该进行的 或操作；对于荧光图像来说其实是 与操作
        # 这里像素点值为 1 表示三通道荧光的交集，像素点为 0 表示三通道背景像素
        bg_mould = background_base[:, :, 0] * background_base[:, :, 1] * background_base[:, :, 2]
        bg_mould[bg_mould > 0] = 1
        bg_mould[bg_mould < 1] = 0
        # 图像保存
        PictureProcess.save_image(result_dir, bg_mould, 'bg_mould.jpg', 'gray')
        # 将扣去的背景和原始图像交集，得到仅有荧光像素的背景，而且这荧光像素点还是三个通道都有值的
        for i in range(3):
            no_background_channels[:, :, i] = original_channels[:, :, i] * bg_mould
        return no_background_channels, bg_mould

    def calculate_Ed_and_Rc(self, no_background_image, background_base_mould):
        """
        计算每个像素点的效率浓度比
        这里应该优化一下，不应该全部像素点进行计算
        :param no_background_image: 去背景后三通道强度矩阵
        :param background_base_mould: 通道共有背景模板
        :return: 效率矩阵, 浓度比矩阵
        """
        IDD = no_background_image[:, :, 0]
        IDA = no_background_image[:, :, 1]
        IAA = no_background_image[:, :, 2]
        F = IDA - self.a * (IAA - self.c * IDD) - self.d * (IDD - self.b * IAA)
        E = (F / (F + self.G * IDD + 1e-12)) * background_base_mould
        Rc = ((self.k * IAA) / (IDD + F / self.G + 1e-12)) * background_base_mould
        return E, Rc

    def computer_subdir(self, main_dir, subdir_name):
        """
        :param main_dir: 主目录名称
        :param subdir_name: 子目录名称
        :return: 处理单个三通道的时间
        """
        _start_time = datetime.now()
        # 判断文件格式是否正确
        curr_subdir_path = os.path.join(main_dir, subdir_name)
        if (isEqual(subdir_name, '.')
                or isEqual(subdir_name, '..')
                or not os.path.isdir(curr_subdir_path)):
            return
        # 创建目录进行结果保存
        result_dir = create_file(curr_subdir_path)
        # 读取三通道图像
        [dd, da, aa] = PictureProcess.read_image(curr_subdir_path)
        # 将三通道图像合并
        image = FRET.build_image_matrix(dd, da, aa)
        # 初始化背景板
        background_base = FRET.initialize_background(image.shape)
        # 计算背景模板 保存背景模板
        no_background_image, background_base_mould = self.calculate_background(image, background_base, result_dir)
        # 计算效率浓度比
        Ed, Rc = self.calculate_Ed_and_Rc(no_background_image, background_base_mould)
        # 由于伪彩图生成很缓慢，所以这里进行判断，加快执行速度
        if self.pcolor:
            # 绘制ED的伪彩图像
            PictureProcess.figure_gray_pic_and_pcolor_pic(Ed, result_dir, "ED")
            # 绘制浓度比的伪彩图
            PictureProcess.figure_gray_pic_and_pcolor_pic(Rc, result_dir, "Rc")
        # 绘制ED的直方图，得到去除极大极小的矩阵
        processed_Ed = PictureProcess.figure_hist(Ed, result_dir, "ED", 0, 0.8, 0.05, 1)

        # 绘制Ec的直方图，得到去除极大极小的矩阵
        processed_Rc = PictureProcess.figure_hist(Rc, result_dir, "Rc", 0, 20, 0.1, 20)
        if processed_Rc is not None and processed_Ed is not None:
            # 计算Ed的均值和标准差
            E_mean, E_std = calculate_mean_and_std(processed_Ed)
            # 计算浓度比的均值和标准差
            Rc_mean, Rc_std = calculate_mean_and_std(processed_Rc)
            # 记录均值和标准差信息
            write_data(result_dir, E_mean, E_std, Rc_mean, Rc_std)
        else:
            print(f"文件{subdir_name}对应的Rc和Ed不存在值")

        return f"文件{subdir_name}处理时间 cost_time={cost_time(_start_time)}s"

    @staticmethod
    def call_back(res):
        """
        线程结束执行函数
        :param res: 返回信息
        """
        FRET.pbar.update(1)
        print(f'执行 {res}')

    @staticmethod
    def err_call_back(err):
        """
        线程调用报错
        :param err: 错误信息
        """
        print(f'出错啦~ error：{str(err)}')

    def start(self, main_dir: str = ''):
        """
        开始函数
        :param main_dir: 处理主目录
        """
        if main_dir == '':
            main_dir = self.main_dir
        self.subdir_list, self.subdir_len = get_path(main_dir)
        # 添加进程池
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        print(f"调用本地电脑核心数量有 use_cpu_num= {multiprocessing.cpu_count() - 1}")
        _start_time = datetime.now()
        # 遍历所有子目录，进行处理
        FRET.pbar = tqdm(total=self.subdir_len)
        for i in range(self.subdir_len):
            pool.apply_async(self.computer_subdir,
                             args=(main_dir, str(self.subdir_list[i])),
                             callback=self.call_back,
                             error_callback=self.err_call_back
                             )

        # self.computer_subdir(main_dir, self.subdir_list[i], pbar)
        # 关闭进程池，不再接受新的进程
        pool.close()
        # 主进程阻塞等待子进程的退出
        pool.join()
        print("执行总时间为：" + str(cost_time(_start_time)) + "s")


if __name__ == "__main__":
    print("测试流程")
    fret = FRET()
    fret.start(r"E:\20231125\ctrl")
    # fret.computer_subdir(r"E:\20231117\20231117hepg2CY\C1Y2", "1")
