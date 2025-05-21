import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设定数据路径
dataset_type = 'fold1'
train_csv_path = f"E:/AMR/DA/Projekt/data/{dataset_type}_list.csv"

# 读取数据
df = pd.read_csv(train_csv_path)

# 统计每个鸟类的不同 vocalization 样本数
bird_vocalization_counts = df.groupby(["bird_name", "vocalization"]).size().unstack(fill_value=0)

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 6))

# 生成不同 vocalization 的颜色
vocalizations = bird_vocalization_counts.columns
x = np.arange(len(bird_vocalization_counts))  # X 轴索引
width = 0.15  # 柱子宽度

# 逐个 vocalization 绘制柱状图
for i, voc in enumerate(vocalizations):
    ax.bar(x + i * width, bird_vocalization_counts[voc], width=width, label=voc)

# 设置 X 轴标签
ax.set_xticks(x + (len(vocalizations) - 1) * width / 2 + 0.2)
ax.set_xticklabels(bird_vocalization_counts.index, rotation=0, ha="right")

# 在每个 bird name 之间添加竖线
for i in range(1, len(x)):
    ax.axvline(x=x[i] - (width / 2) - 0.1, color='gray', linestyle='--', linewidth=1)

# 添加标题和标签
ax.set_xlabel("Bird Name")
ax.set_ylabel("Sample Count")
ax.set_title(f"{dataset_type} Dataset Distribution")
ax.legend(title="Vocalization")

# 显示网格
ax.grid(axis="y", linestyle="--", alpha=0.7)

# 保存图像
# plt.tight_layout()
# plt.savefig("E:/AMR/DA/Projekt/experiments/exp1/data_distribution.png")
# print("📊 数据分布图已保存: E:/AMR/DA/Projekt/experiments/exp1/data_distribution.png")

# 显示图像
plt.show()
