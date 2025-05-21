import os

# 设置音频文件目录
AUDIO_FILES_DIR = "E:/AMR/DA/Projekt/data/Audio_files_ori"
root = 'E:/AMR/DA/Projekt/bird_cls_cnn/data_preprocess'
# 存储鸟类名称的集合（自动去重）
bird_species = set()

# 遍历 Audio_files 目录中的所有文件夹
for folder_name in os.listdir(AUDIO_FILES_DIR):
    folder_path = os.path.join(AUDIO_FILES_DIR, folder_name)

    # 确保是文件夹
    if os.path.isdir(folder_path):
        # 解析鸟类名称（去掉前缀）
        if " - " in folder_name:
            bird_name = folder_name.split(" - ", 1)[1]  # 只保留 "-" 之后的鸟类名
            bird_species.add(bird_name)

# **按字母排序，生成有序列表**
bird_list = sorted(bird_species)

# **生成字典映射**
bird_name_mapping = {bird: idx for idx, bird in enumerate(bird_list)}

# **格式化 Python 代码**
formatted_list = "selected_birds = [\n" + "\n".join(f'    "{bird}",' for bird in bird_list) + "\n]\n"
formatted_dict = "bird_name_mapping = {\n" + "\n".join(f'    "{bird}": {idx},' for bird, idx in bird_name_mapping.items()) + "\n}"

# **合并并写入 `selected_birds.py`**
output_path = os.path.join(root, "selected_birds.py")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(formatted_list + "\n" + formatted_dict + "\n")

# **打印输出**
print("\n📋 训练数据集中的鸟类：")
print(formatted_list)
print("\n📋 鸟类索引映射：")
print(formatted_dict)

print(f"\n✅ 鸟类名称和映射已保存到 {output_path}")