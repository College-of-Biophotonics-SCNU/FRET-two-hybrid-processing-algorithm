import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sympy.testing.runtests import method

from E_FRET_gpu.extraction.phenotype_feature_extraction import PFeatureExtraction


class LocalPCAFeatureExtraction(PFeatureExtraction):
    """
    局部PCA算法，利用每组不同的pca进行计算得出结果
    """
    def __init__(self, path, n_components=10, save_root=None):
        super().__init__(path)
        # control 组主成分均值
        self.principal_component_average = {}
        # 主成分数量
        self.n_components = n_components
        # 文件保存路径
        self.save_root = save_root
        if save_root is None:
            self.save_root = self.directory
        # 划分时间后的单视野统计
        self.divide_time_site_data_pd = None

    def start(self, need_pretreatment=False, remove_outliers_method=''):
        """
        具体流程
        1. 特征筛选预处理，留下有效特征
        2. pca降维
        3. 计算control组的主成分均值
        4. 计算表型表证值
        5. 计算结果保存
        """
        ########################################
        ## 特征筛选
        ########################################
        super().start(need_pretreatment)
        # 提取特征和标签
        X = self.data_pd[self.feature_columns].values
        # 进行预处理流程 TODO 预处理效果存在问题
        X = self.remove_outliers(X, method=remove_outliers_method)
        y = self.data_pd['Metadata_Label'].values

        # 开始使用不同时间点的对照组进行计算过程
        result_df = pd.DataFrame()
        # 按 Metadata_hour 分组，并对每个分组单独处理
        grouped = self.data_pd.groupby('Metadata_hour')
        # 将 PCA 结果转换为 DataFrame
        principal_components = [f'Principal_component_{i}' for i in range(self.n_components)]
        # 添加新降维特征到特征列表名称中
        self.feature_columns = self.feature_columns + principal_components

        min_max_scaler = MinMaxScaler()
        # 创建对象
        for treatment in self.treatments:
            # 计算每个 control 不同时间的特征均值
            self.principal_component_average[treatment] = {}
        for hour, group in grouped:
            ########################################
            ## 特征降维 分为划分时间以及不划分时间的情况
            ########################################
            features, weights = self.reduce(X[group.index], self.n_components)
            pca_df = pd.DataFrame(features, columns=principal_components, index=group.index)
            # 拼接原始数据和 PCA 结果
            group = pd.concat([group, pca_df], axis=1)

            ########################################
            ## control主成分按照不同时间节点均值计算
            ########################################
            for treatment in self.treatments:
                # 计算每个 control 不同时间的特征均值
                self.principal_component_average[treatment][hour] = {}
                treatment_pd = group[group['Metadata_treatment'] == treatment]
                for component in range(0, self.n_components):
                    self.principal_component_average[treatment][hour][component] = treatment_pd[f'Principal_component_{component}'].mean()

            ########################################
            ## 特征公式计算，同时将计算结果进行保存到pandas的df中
            ########################################
            # 计算每个分组中control数值
            result_values = self.function(features, weights=weights, control_component=self.principal_component_average['control'][hour])
            # 每个result_values 需要进行归一化操作
            result_values_rescaler = min_max_scaler.fit_transform(result_values)
            # 对于局部PCA的特征需要进行归一化操作
            group[self.phenotypic_value_name] = result_values_rescaler

            result_df = pd.concat([result_df, group], ignore_index=True)

        self.data_pd = result_df

        ########################################
        ## 按照每个site进行统计分析
        ########################################
        compute_columns = self.feature_columns + [self.phenotypic_value_name]
        # 按 Metadata_site 和 Metadata_hour 分组，并计算每个分组中特征的均值
        self.divide_time_site_data_pd = self.data_pd.groupby(['Metadata_site', 'Metadata_hour', 'Metadata_treatment'])[compute_columns].mean().reset_index()

        ########################################
        ## 输出保存展示最后的结果 根据时间划分的局部PCA算法
        ########################################
        # 单细胞尺度的统计图F
        self.show(self.data_pd,
                  self.phenotypic_value_name,
                  'Metadata_Label',
                  'faceted_box',
                  save_path=os.path.join(self.save_root),
                  save_name=self.experiment_name + '_Local_PCA_Cell_S_value')
        self.save(self.data_pd,
                  save_path=self.save_root,
                  save_name='Local_PCA_Cell_proceed')
        # 单视野尺度统计
        self.show(self.divide_time_site_data_pd,
                  self.phenotypic_value_name,
                  'Metadata_Label',
                  'faceted_box',
                  save_path=os.path.join(self.save_root),
                  save_name=self.experiment_name + '_Local_PCA_Site_S_value')
        self.save(self.divide_time_site_data_pd,
                  save_path=self.save_root,
                  save_name='Local_PCA_Site_proceed')

    def reduce(self, X, n_components=10):
        # 特征归一化
        standard_scaler = StandardScaler()
        min_max_scaler = StandardScaler()
        X_scaled = standard_scaler.fit_transform(X)
        # 创建PCA对象，并设置要保留的主成分数目
        pca = PCA(n_components=n_components)
        # 将最后的降维得特征向量进行归一化 方便后续计算
        features = pca.fit_transform(X_scaled)
        features = min_max_scaler.fit_transform(features)
        # 获取每个主成分的方差贡献率
        weights = pca.explained_variance_ratio_
        return features, weights

    def function(self, features, weights, control_component=None):
        # 最后的表型表证值
        S_value = np.zeros((features.shape[0], 1))
        for component in range(0, self.n_components):
            # 计算每个样本在当前主成分上的差异，并取绝对值
            diff = weights[component] * np.abs(features[:, component] / (control_component[component] + 1e-10) - 1)
            # 将差异值累加到 S_value 中
            S_value += diff.reshape(-1, 1)
        return S_value
