import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r'/data/csv/20240610_A549_EGFR-FRET_Protein.csv'
df = pd.read_csv(file_path)

# 提取特征和标签
metadata_columns = [col for col in df.columns if col.startswith('Metadata_')]
metadata_columns.extend(['ImageNumber', 'ObjectNumber'])

# 特征列
feature_columns = [col for col in df.columns if col not in metadata_columns]

# 标签列
df['Label'] = df['Metadata_treatment'] + '_' + df['Metadata_hour'].astype(str)

# 提取特征和标签
X = df[feature_columns].values
y = df['Label'].values

# 特征归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 进行PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 创建DataFrame以方便绘图
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Label'] = y

# 绘制PCA图像
plt.figure(figsize=(10, 8))
for label in np.unique(y):
    subset = pca_df[pca_df['Label'] == label]
    plt.scatter(subset['PC1'], subset['PC2'], label=label)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Normalized Features with Labels')
plt.legend()
plt.show()