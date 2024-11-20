import warnings
import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


warnings.filterwarnings("ignore", category=UserWarning, module="umap.umap_")

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

# 进行UMAP降维
umap_model = umap.UMAP(n_components=3, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# 创建DataFrame以方便绘图
umap_df = pd.DataFrame(data=X_umap, columns=['UMAP1', 'UMAP2', 'UMAP3'])
umap_df['Label'] = y

# 绘制3D UMAP图像
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for label in np.unique(y):
    subset = umap_df[umap_df['Label'] == label]
    ax.scatter(subset['UMAP1'], subset['UMAP2'], subset['UMAP3'], label=label)

ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')
ax.set_zlabel('UMAP Component 3')
ax.set_title('3D UMAP of Normalized Features with Labels')
ax.legend()
plt.show()