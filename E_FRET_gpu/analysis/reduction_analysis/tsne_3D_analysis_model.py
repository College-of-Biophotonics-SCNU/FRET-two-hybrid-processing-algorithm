import warnings

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.context")

# 读取CSV文件
file_path = r'/data/csv/20240803_MCF7_BF.csv'
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

# 进行t-SNE降维
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 创建DataFrame以方便绘图
tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2', 'TSNE3'])
tsne_df['Label'] = y

# 绘制3D t-SNE图像
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for label in np.unique(y):
    subset = tsne_df[tsne_df['Label'] == label]
    ax.scatter(subset['TSNE1'], subset['TSNE2'], subset['TSNE3'], label=label)

ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
ax.set_title('3D t-SNE of Normalized Features with Labels')
ax.legend()
plt.show()