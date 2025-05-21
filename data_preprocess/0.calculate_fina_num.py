import os

# 设置存储音频文件的根目录
ROOT_DOWNLOAD_FOLDER = "E:/AMR/DA/Projekt/data/Audio_files_ori"

# 统计结果存储
folder_counts = {}

# 遍历根目录下的所有文件夹
for folder_name in os.listdir(ROOT_DOWNLOAD_FOLDER):
    folder_path = os.path.join(ROOT_DOWNLOAD_FOLDER, folder_name)

    # 确保是文件夹
    if os.path.isdir(folder_path):
        # 获取该文件夹中的所有文件
        file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        
        # 记录到字典
        folder_counts[folder_name] = file_count

# **打印统计结果**
print("\n📊 音频文件统计结果：")
for folder, count in folder_counts.items():
    print(f"📁 {folder}: {count} 个文件")

# # **如果你希望以 CSV 格式导出**
# import csv

# csv_path = os.path.join(ROOT_DOWNLOAD_FOLDER, "audio_file_counts.csv")
# with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Folder Name", "File Count"])  # 表头
#     for folder, count in folder_counts.items():
#         writer.writerow([folder, count])
# 
# print(f"\n✅ 统计数据已保存到 {csv_path}")
