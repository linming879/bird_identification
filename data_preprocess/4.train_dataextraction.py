import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 📌 文件路径
data_csv_path = "E:/AMR/DA/Projekt/data/all_data_meta.csv"  # 原始数据
train_csv_path = "E:/AMR/DA/Projekt/data/train_list.csv"  # 训练集
test_csv_path = "E:/AMR/DA/Projekt/data/test_list.csv"  # 测试集

# 🎯 目标鸟类（仅保留这些类别）
selected_birds = [
    "Black-headed Gull", "Canada Goose", "Carrion Crow", "Common Blackbird",
    "Common Chaffinch", "Common Kingfisher", "Common Redstart", "Common Wood Pigeon",
    "Dunnock", "Eurasian Blackcap", "Eurasian Blue tit", "Eurasian Bullfinch",
    "Eurasian Coot", "Eurasian Golden Oriole", "Eurasian Jay", "Eurasian Nuthatch",
    "Eurasian Siskin", "Eurasian Treecreeper", "Eurasian Wren", "European Goldfinch",
    "European Robin", "Goldcrest", "Great Spotted Woodpecker", "Great Tit",
    "Hawfinch", "Hooded Crow", "Lesser Black-backed Gull", "Long-tailed Tit",
    "Mallard", "Marsh Tit", "Redwing", "Rook", "Short-toed Treecreeper", "Stock Dove",
]

# 📌 读取数据
df = pd.read_csv(data_csv_path)
print(f"📊 原始数据: {len(df)} 样本")

# 仅保留目标鸟类
df = df[df["bird_name"].isin(selected_birds)]

# 提取音频编号（去掉 segX，只保留 `XCxxxxx`）
df["audio_id"] = df["number"].apply(lambda x: x.split("_")[0] if "_" in x else x)

# 🚀 **按音频编号进行划分**，保证所有 `segX` 来自同一个 `XCxxxxx` 文件的片段不会被分开
unique_audio_ids = df["audio_id"].unique()
train_audio_ids, test_audio_ids = train_test_split(unique_audio_ids, test_size=0.2, random_state=2024, shuffle=True)

# 🚀 按 `audio_id` 分配数据
train_df = df[df["audio_id"].isin(train_audio_ids)]
test_df = df[df["audio_id"].isin(test_audio_ids)]

# 确保目标文件夹存在
os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)

# 📌 保存训练 & 测试集
train_df.drop(columns=["audio_id"], inplace=True)
test_df.drop(columns=["audio_id"], inplace=True)

train_df.to_csv(train_csv_path, index=False, encoding="utf-8")
test_df.to_csv(test_csv_path, index=False, encoding="utf-8")

print(f"✅ 训练集已保存至: {train_csv_path}, 样本数: {len(train_df)}")
print(f"✅ 测试集已保存至: {test_csv_path}, 样本数: {len(test_df)}")
