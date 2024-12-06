import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r'D:\data\qrm\EGFR_CFP+GRB2_YFP\2024.06.11_A549_SN_8H' + '//Ed_features.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 检查是否有缺失值并处理（可选）
if df.isnull().values.any():
    print("Warning: There are missing values in the data.")
    # 例如，可以选择删除包含缺失值的行
    df.dropna(inplace=True)

# 确保所有需要的列都在 DataFrame 中
required_columns = ['Metadata_treatment', 'Ed_cell_mean_value', 'Ed_agg_mean_value', 'Ed_agg_top_50_value', 'Ed_agg_top_25_value']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"The following columns are missing from the DataFrame: {missing_columns}")

# 设置Seaborn样式
sns.set(style="whitegrid")

# 创建一个图形和子图
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()  # 将2x2的数组展平为一维数组以便迭代

# 定义要绘制的特征列表
features = ['Ed_cell_mean_value', 'Ed_agg_mean_value', 'Ed_agg_top_50_value', 'Ed_agg_top_25_value']

# 绘制每个特征的箱型图
for ax, feature in zip(axes, features):
    sns.boxplot(x='Metadata_treatment', y=feature, data=df, ax=ax)
    ax.set_title(f'Boxplot of {feature} by Metadata_treatment')
    ax.set_xlabel('Metadata_treatment')
    ax.set_ylabel(feature)

# 调整布局以防止重叠
plt.tight_layout()

# 显示图形
plt.show()