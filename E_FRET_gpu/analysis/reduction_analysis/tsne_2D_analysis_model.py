import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r'C:\Code\python\FRET-two-hybrid-processing-algorithm\data\csv\20240610_A549_EGFR-FRET_Protein.csv'
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
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 创建DataFrame以方便绘图
tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['Label'] = y

# 绘制t-SNE图像
plt.figure(figsize=(10, 8))
for label in np.unique(y):
    subset = tsne_df[tsne_df['Label'] == label]
    plt.scatter(subset['TSNE1'], subset['TSNE2'], label=label)

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Normalized Features with Labels')
plt.legend()
plt.show()