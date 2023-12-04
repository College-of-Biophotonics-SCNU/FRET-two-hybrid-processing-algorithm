import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Fret:
    """
    FRET计算 gaolu
    """
    def __init__(self,
                 a: int = 0.177327,
                 b: int = 0.00997,
                 c: int = 0.001244,
                 d: int = 0.791588,
                 G: int = 7.735335,
                 k: int = 0.538395,
                 exposetimes: tuple = (300, 300, 300)
                 ):
        """
        :param a: 校正因子a
        :param b: 校正因子b
        :param c: 校正因子c
        :param d: 校正因子d
        :param G: 校正因子G
        :param k: 校正因子k
        :param exposetimes: 三通道曝光时间
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.G = G
        self.k = k
        self.exposetimes = exposetimes

    @staticmethod
    def get_path(
                 maindirpath: str,
                 ):
        """
        待处理文件夹路径获取
        :param maindirpath: 待处理文件夹路径
        :return: 文件列表，文件数量
        """
        subdir = os.listdir(maindirpath)
        length = len(subdir)
        return subdir, length

    @staticmethod
    def isEqual(num1, num2):
        """
        比较判断
        :param num1: 比较参数1
        :param num2: 比较参数2
        :return: 比较结果
        """
        if num1 < num2:
            return False
        if num1 > num2:
            return False
        if num1 == num2:
            return True

    @staticmethod
    def creat_file(
                   currSubdirpath: str,
                   ):
        """
        创建结果目录
        :param currSubdirpath: 子文件夹路径
        :return: 保存处理结果路径
        """
        resultdirpath = os.path.join(currSubdirpath, 'E-FRET results1')
        # 结果目录不存在则创建
        if not os.path.isdir(resultdirpath):
            os.makedirs(resultdirpath)
        return resultdirpath

    @staticmethod
    def read_image(currSubdirpath: str):
        """
        读取子文件夹中的三通道图片
        :param currSubdirpath: 子文件路径
        :return: 三通道图像数组
        """
        dd = mpimg.imread(os.path.join(currSubdirpath, 'DD.tif'))
        da = mpimg.imread(os.path.join(currSubdirpath, 'DA.tif'))
        aa = mpimg.imread(os.path.join(currSubdirpath, 'AA.tif'))
        return dd, da, aa

    @staticmethod
    def initialize_BG(
                      M,
                      ):
        """
        初始化背景矩阵
        :param M: 背景矩阵维度
        :return: 背景基地，背景，背景模板
        """
        shape = (M[0], M[1], 3)
        dtype = np.double
        BGjidi = np.ones(M)
        Bg = np.ones(shape, dtype=dtype)
        BGmoban = Bg
        return BGjidi, Bg, BGmoban

    @staticmethod
    def diimage(M,
                dd,
                da,
                aa):
        """
        构建三通道图像矩阵
        :param M: 矩阵维度
        :param dd: DD通道图像
        :param da: DA通道图像
        :param aa: AA通道图像
        :return: 三通道图像矩阵
        """
        shape = (M[0], M[1], 3)
        dtype = np.double
        I00 = np.ones(shape, dtype=dtype)
        I00[:, :, 0] = dd
        I00[:, :, 1] = da
        I00[:, :, 2] = aa
        I11 = I00
        return I00, I11

    def BG_calculate(self,
                     I00,
                     I11,
                     BGjidi,
                     Bg,
                     BGmoban):
        """
        分别确定三个通道的背景
        :param I00: 初始三通道强度矩阵
        :param I11: 存储去背景后的强度矩阵
        :param BGjidi: 背景基底矩阵
        :param Bg: 背景值
        :param BGmoban: 三个通道背景模板
        :return: 去背景后强度矩阵，三通道背景模板矩阵
        """
        for j in range(3):
            Ii = I00[:, :, j]
            n = np.arange(1, 2000, 1)
            Ii = Ii[Ii > 0]
            Ii = Ii[Ii < 4000]   # 假定背景所在范围
            vddgz, Iandcolumn, patches = plt.hist(Ii, n)
            plt.close()
            DD1gz = max(vddgz)
            colDD = np.where(vddgz == DD1gz)
            Bg[:, :, j] = BGjidi*np.max(colDD)
            I11[:, :, j] = I00[:, :, j]-Bg[:, :, j]
            BGmoban[:, :, j] = I00[:, :, j]-Bg[:, :, j]*1.5  # 模板阈值提高值

            BGmoban[BGmoban < 0] = 0
            BGmoban[BGmoban > 50000] = 0
            I11[:, :, j] = np.divide(I1[:, :, j], self.exposetimes[j])  # 对通道曝光时间归一化
        return I11, BGmoban

    @staticmethod
    def get_moban(
                  BGmoban,
                  resultdirpath):
        """
        计算三通道共有模板
        :param BGmoban: 三通道每个通道的模板 (2048,2048,3)
        :param resultdirpath: 模板的保存路径
        :return: 三通道共有模板矩阵 (2048,2048)
        """
        moban1 = BGmoban[:, :, 0]*BGmoban[:, :, 2]
        Moban = moban1*BGmoban[:, :, 1]
        Moban[Moban > 0] = 1
        Moban[Moban < 1] = 0
        mobandir = os.path.join(resultdirpath, 'moban.jpg')
        plt.imsave(mobandir, Moban, cmap='gray')
        return Moban

    def calculate_EdRc(self,
                       I11,
                       Moban):
        """
        计算每个像素点的效率浓度比
        :param I11: 去背景后三通道强度矩阵
        :param Moban: 通道共有背景模板
        :return: 效率矩阵和浓度比矩阵
        """
        IDD = I11[:, :, 0]
        IDA = I11[:, :, 1]
        IAA = I11[:, :, 2]
        F = IDA-self.a*(IAA-self.c*IDD)-self.d*(IDD-self.b*IAA)
        E1 = (F/(F+self.G*IDD+1e-12))*Moban
        Rc1 = ((self.k*IAA)/(IDD+F/self.G+1e-12))*Moban
        return E1, Rc1

    @staticmethod
    def figure_ED(E1, resultdirpath):
        """
        绘制效率和效率伪彩图
        :param E1:效率矩阵
        :param resultdirpath: 保存路径
        """

        Edir = os.path.join(resultdirpath, 'ED.jpg')
        plt.imsave(Edir, E1, cmap='gray')
        E1 = np.flipud(E1)
        E11wei = E1
        imgGcf = plt.figure()
        plt.pcolor(E11wei)
        plt.axis('off')
        plt.title('EDwei')
        plt.colorbar()
        plt.close()
        Eweidir = os.path.join(resultdirpath, 'EDwei.jpg')
        imgGcf.savefig(Eweidir)

    @staticmethod
    def figure_EDhist(
                      E1,
                      resultdirpath):
        """
        绘制效率统计图
        :param E1: 效率矩阵
        :param resultdirpath: 存储路径
        :return: 去除极大极小值的效率矩阵
        """
        ae = E1[E1 > 0]
        ae = ae[ae < 1]
        bE = np.arange(0, 0.8, 0.05)
        m1, n1, pathches = plt.hist(ae, bE, color='black')
        imgGcf = plt.figure()
        maxAVG = max(m1)
        m2 = m1 / maxAVG
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.ylabel('frequency', size=20)
        n2 = np.arange(0, 0.75, 0.05)
        plt.bar(n2, m2, width=0.04, color='black')
        plt.close()
        Ehistdir = os.path.join(resultdirpath, 'EDhist.jpg')
        imgGcf.savefig(Ehistdir)
        return ae

    @staticmethod
    def calculate_avED(ae):
        """
        计算效率均值和标准差
        :param ae: 效率矩阵
        :return: 效率均值，标准差
        """
        ave = np.mean(ae)
        esd = np.std(ae, ddof=1)
        return ave, esd

    @staticmethod
    def figure_Rc(
                  R11,
                  resultdirpath):
        """
        绘制浓度比伪彩图
        :param R11: 浓度比矩阵
        :param resultdirpath: 保存文件路径
        """
        Rdir = os.path.join(resultdirpath, 'Rc.jpg')
        plt.imsave(Rdir, R11, cmap='gray')
        R11 = np.flipud(R11)
        R11wei = R11
        imgGcf = plt.figure()
        plt.pcolor(R11wei)
        plt.axis('off')
        plt.title('Rcwei')
        plt.colorbar()
        plt.close()
        Rcweidir = os.path.join(resultdirpath, 'Rcwei.jpg')
        imgGcf.savefig(Rcweidir)

    @staticmethod
    def figure_Rchist(
                      R11,
                      resultdirpath):
        """
        绘制浓度比统计图
        :param R11: 浓度比矩阵
        :param resultdirpath:保存图像路径
        :return: 浓度比矩阵
        """
        ar = R11[R11 > 0]
        ar = ar[ar < 20]
        bR = np.arange(0, 20, 0.1)
        plt.figure()
        m1, n1, pathches = plt.hist(ar, bR, color='black')
        plt.close()
        imgGcf = plt.figure()
        maxAVG = max(m1)
        m2 = m1 / maxAVG
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.ylabel('frequency', size=20)
        n2 = np.arange(0, 19.9, 0.1)
        plt.bar(n2, m2, width=0.08, color='black')
        plt.close()
        Rchistdir = os.path.join(resultdirpath, 'Rchist.jpg')
        imgGcf.savefig(Rchistdir)
        return ar

    @staticmethod
    def calculate_avRc(ar):
        """
        计算浓度比均值和标准差
        :param ar: 浓度比矩阵
        :return: 浓度比均值，标准差
        """
        avr = np.mean(ar)
        rsd = np.std(ar, ddof=1)
        return avr, rsd

    @staticmethod
    def write_data(
                   resultdirpath,
                   ave,
                   esd,
                   avr,
                   rsd):
        """
        将均值标准差数据写入txt文件
        :param resultdirpath:文件路径
        :param ave: 效率均值
        :param esd:效率标准差
        :param avr: 浓度比均值
        :param rsd: 浓度比标准差
        """
        calculatdir = os.path.join(resultdirpath, 'calculateResult.txt')
        file = open(calculatdir, 'w')
        file.write('avE:')
        file.write(str(ave))
        file.write('\nEsd:')
        file.write(str(esd))
        file.write('\navR:')
        file.write(str(avr))
        file.write('\nRsd:')
        file.write(str(rsd))
        file.close()
        print('计算结果已保存')










edrc = Fret()
maindir = 'E:/20231125/sora25'
s, l, = edrc.get_path(maindir)
index = 0
for i in tqdm(range(l)):
    currSubdirPath = os.path.join(maindir, s[i])
    if edrc.isEqual(s[i], '.') or edrc.isEqual(s[i], '..') or not os.path.isdir(currSubdirPath):
        continue
    resultdir = edrc.creat_file(currSubdirPath)  # 创建处理结果目录
    [DD, DA, AA] = edrc.read_image(currSubdirPath)
    m = DD.shape
    [I0, I1] = edrc.diimage(m, DD, DA, AA)
    [BGjidigz, BG, BGmb] = edrc.initialize_BG(m)
    [I1, BGmb] = edrc.BG_calculate(I0, I1, BGjidigz, BG, BGmb)
    moban = edrc.get_moban(BGmb, resultdir)
    [E11, Rc11] = edrc.calculate_EdRc(I1, moban)
    # edrc.figure_ED(E11, resultdir)
    aE = edrc.figure_EDhist(E11, resultdir)
    [avE, Esd] = edrc.calculate_avED(aE)
    # edrc.figure_Rc(Rc11, resultdir)
    aR = edrc.figure_Rchist(Rc11, resultdir)
    [avR, Rsd] = edrc.calculate_avRc(aR)
    edrc.write_data(resultdir, avE, Esd, avR, Rsd)
