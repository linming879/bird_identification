import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 设定文件路径
data_csv_path = "E:/AMR/DA/Projekt/data/all_data_meta.csv"  # 原始数据
fold1_csv_path = "E:/AMR/DA/Projekt/data/fold1_list.csv"  # 第一折
fold2_csv_path = "E:/AMR/DA/Projekt/data/fold2_list.csv"  # 第二折

# 设定要筛选的鸟类
selected_birds = {"Common Blackbird", "Eurasian Blue tit", "Great Tit", "Short-toed Treecreeper"}

# 读取 all_data_meta.csv
df = pd.read_csv(data_csv_path)
print(f"总数据集信息: {data_csv_path}, 样本数: {len(df)}")
# 过滤指定的鸟类
df = df[df["bird_name"].isin(selected_birds)]

# 确保目标文件夹存在
os.makedirs(os.path.dirname(fold1_csv_path), exist_ok=True)

# 存储两个折的数据
fold1_data = []
fold2_data = []

# 按 `bird_name` 和 `vocalization` 进行独立划分
for (bird, vocalization), group in df.groupby(["bird_name", "vocalization"]):
    # 按 50% 划分数据（交叉验证两折）
    fold1_subset, fold2_subset = train_test_split(group, test_size=0.5, random_state=2024, shuffle=True)
    
    # 存储数据
    fold1_data.append(fold1_subset)
    fold2_data.append(fold2_subset)

# 合并所有数据
fold1_df = pd.concat(fold1_data)
fold2_df = pd.concat(fold2_data)

# 保存交叉验证的两折数据
fold1_df.to_csv(fold1_csv_path, index=False, encoding="utf-8")
fold2_df.to_csv(fold2_csv_path, index=False, encoding="utf-8")

print(f"第一折训练集已保存至: {fold1_csv_path}, 样本数: {len(fold1_df)}")
print(f"第二折训练集已保存至: {fold2_csv_path}, 样本数: {len(fold2_df)}")
