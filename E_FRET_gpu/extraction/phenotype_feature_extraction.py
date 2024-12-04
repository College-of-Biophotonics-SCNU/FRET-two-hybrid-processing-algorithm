"""
基于CellProfiler的明场特征提取数据进行特征融合
1. 使用 PCA 降维算法进行降维分析，融合单个特征值

该模块需要使用模块化进行分析
1. 使用了父子类抽象继承
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from E_FRET_gpu.tool.draw_plt import draw_frequency_histogram, draw_faceted_box_plot

class PFeatureExtraction:
    """
    明场特征提取父类
    """
    def __init__(self, path, phenotypic_value_name='S_value'):
        """
        :param path: 文件读取的路径
        :param phenotypic_value_name: 表征表征的名字
        """
        # csv 文件存放地址
        self.path = path
        # 表征表征值赋值
        self.phenotypic_value_name = phenotypic_value_name
        # 获取文件存放路径
        # 获取文件的目录和文件名（带后缀）
        self.directory, _ = os.path.split(path)
        # 文件拓展名字
        self.file_extension = os.path.splitext(path)[1].lower()
        # 获取文件名称 作为实验批次名称
        self.experiment_name = os.path.splitext(os.path.basename(path))[0]
        # 排除的特征值
        self.exclude_features = ['Metadata_', 'ImageNumber', 'ObjectNumber']
        # 有效的特征名称
        self.feature_columns = []
        # 最后的表型表证值
        self.result_values = None
        # 对应的 Metadata_Label 标签
        self.result_labels = None
        # 读取文件
        if self.file_extension == '.csv':
            # 读取CSV文件
            df = pd.read_csv(path)
        elif self.file_extension == '.xlsx':
            # 读取Excel文件
            df = pd.read_excel(path, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported file type: {path}")
        # pandas 数据进行保存
        self.data_pd = df
        # 统计所有不同时间点
        self.hours = self.data_pd['Metadata_hour'].unique()
        # 统计不同实验组
        self.treatments = self.data_pd['Metadata_treatment'].unique()
        # z-score 方法阈值
        self.z_score_threshold = 3

    def start(self, need_pretreatment=False):
        """
        算法启动开始流程
        :param need_pretreatment: 查看是否需要进行特征合并操作
        """
        # 除去不需要的特征值
        self.feature_columns = [col for col in self.data_pd.columns if
                           all(feature not in col for feature in self.exclude_features)]
        # 标签列
        self.data_pd['Metadata_Label'] = self.data_pd['Metadata_treatment'] + '_' + self.data_pd['Metadata_hour'].astype(str)
        if need_pretreatment:
            self.pretreatment()

    def pretreatment(self):
        """
        预处理流程
        1. 主要是对于pandas数据进行处理，对于明场特征存在三个焦平面的数据进行均值统计，按照{}_{}的特征命名规律进行均值计算
        """
        ########################################
        ## 计算特征类均值
        ########################################
        columns = self.feature_columns
        df = self.data_pd
        # 创建字典来存储每组特征的列名
        grouped_columns = {}

        # 遍历所有特征名
        for col in columns:
            # 提取前三个单词作为键（假设特征名使用下划线分隔）
            parts = col.split('_')
            if len(parts) >= 3:
                key = '_'.join(parts[:3])
            else:
                key = col  # 如果特征名不足三个部分，则用原名列名作为键
            # 如果键不存在于字典中，则初始化为空列表
            if key not in grouped_columns:
                grouped_columns[key] = []
            # 将当前特征添加到对应的列表中
            grouped_columns[key].append(col)
        # 计算每组特征的平均值
        new_columns = {}
        for key, cols in grouped_columns.items():
            # 计算每组特征的平均值
            new_columns[f'{key}_mean'] = df[cols].mean(axis=1)
        # 将新的平均值列添加到原始 DataFrame 中
        df_new = df.copy()
        df_new = pd.concat([df_new, pd.DataFrame(new_columns)], axis=1)
        # 删除原始的特征列
        for cols in grouped_columns.values():
            df_new = df_new.drop(columns=cols)
        # 设置新的特征pandas
        self.data_pd = df_new
        # 设置新的特征名称
        self.feature_columns = list(new_columns.keys())


    def reduce(self, processed_pd):
        """
        降维操作
        :param processed_pd: 处理后的 pandas 数据
        """
        pass

    def function(self, features, weights):
        """
        特征融合公式
        """
        pass

    def save(self, save_pd=None, save_path=None, save_name=''):
        """
        保存表型表征值
        :param save_pd: 保存的 pd 文件
        :param save_path: 保存路径
        :param save_name: 保存的名称
        """
        if save_pd is None:
            save_pd = self.data_pd
        if save_path is None:
            save_path = self.directory
        # 保存到源文件
        if self.file_extension == '.csv':
            # 保存CSV文件
            save_pd.to_csv(str(os.path.join(save_path, self.experiment_name + '_' + save_name + '.csv')), index=False)
        elif self.file_extension == '.xlsx':
            # 保存 xlsx 文件
            save_pd.to_excel(str(os.path.join(save_path, self.experiment_name + '_' + save_name + '.xlsx')), index=False)

    def remove_outliers(self, X=None, method=None):
        """
        数据预处理流程
        1. 采用 Z-Score 方法
        2. 采用 iqr 方法
        3. 采用 RobustScaler 方法
        """
        print("原有数据格式变为", self.data_pd.shape)
        # 为空的情况 返回原数据
        if method is None:
            return X
        if method == 'z-score':
            # 计算每列的 Z-Score
            z_scores = (X - X.mean()) / X.std()
            # 找出绝对值大于阈值的行
            outliers = (np.abs(z_scores) > self.z_score_threshold).any(axis=1)
            # 去除异常值 对于原有data_pd进行操作
            self.data_pd = self.data_pd[~outliers]
            self.data_pd = self.data_pd.reset_index(drop=True)
            print("经过预处理后数据格式变为", self.data_pd.shape)
            return X[~outliers]
        elif method == 'iqr':
            # 创建一个掩码数组，初始值为 True，表示所有数据点都保留
            mask = np.ones(X.shape[0], dtype=bool)
            for col_idx in range(X.shape[1]):
                column = X[:, col_idx]
                # 计算四分位数
                Q1 = np.percentile(column, 25)
                Q3 = np.percentile(column, 75)
                # 计算 IQR
                IQR = Q3 - Q1
                # 定义上下限
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # 更新掩码，标记不在上下限之间的数据点为 False
                mask &= (column >= lower_bound) & (column <= upper_bound)
            # 应用掩码，过滤掉异常值
            X_cleaned = X[mask]
            # 还原对应的pandas数据
            self.data_pd = self.data_pd[mask].reset_index(drop=True)
            print("经过预处理后数据格式变为", self.data_pd.shape)
            return X_cleaned
        elif method == 'robust':
            scaler = RobustScaler()
            X_robust_scaled = scaler.fit_transform(X)
            return X_robust_scaled
        else:
            return X

    @staticmethod
    def show(result_pd, result_name, label_name='Metadata_Label', plot_type='hist', save_path=None, save_name=''):
        """
        数据展示
        :param result_pd: 最后的result数据
        :param plot_type: 输出图像的类型，目前有
        :param save_path: 保存路径
        :param result_name: pd中表型表征值的列明场
        :param label_name: pd中标志的列名称
        :param save_path: 图像保存路径
        :param save_name: 保存的文件名称
        1. hist 频率直方图统计
        2. faceted_box 分面箱型图统计，按照时间维度和实验组维度组合统计表型表征值
        3. box 箱型图统计，按照时间维度进行划分，没有多个维度进行统计
        """
        # 示例数据
        if plot_type == 'hist':
            draw_frequency_histogram(result_pd[result_name],
                                     result_pd[label_name],
                                     save_path=save_path,
                                     save_name=save_name)
        elif plot_type == 'faceted_box':
            draw_faceted_box_plot(result_pd,
                                  'Metadata_hour',
                                  result_name,
                                  'Metadata_treatment',
                                  save_path=save_path,
                                  save_name=save_name)