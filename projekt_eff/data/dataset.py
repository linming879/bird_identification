import os
import torch
import numpy as np
import pandas as pd
import albumentations as albu
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import librosa
import albumentations as albu

class BirdCLEFDataset_backup(Dataset):
    def __init__(self, csv_path, bird_name_mapping, augment=False, img_size=(256, 256), model_type="EfficientNet", input_channel=1, use_xymasking=True, max_samples=3000):
        """
        鸟类音频频谱数据集
        :param csv_path: 训练/测试数据的 CSV 文件路径
        :param bird_name_mapping: 字典，bird_name -> label 映射
        :param augment: 是否进行数据增强
        :param img_size: 频谱图目标尺寸
        :param model_type: 选择模型类型 ("EfficientNet", "EfficientNetV2", "ReXNet")
        :param input_channel: 选择输入通道数（1 = 单通道，3 = 复制为三通道）
        :param use_xymasking: 是否使用 XY 遮挡
        :param max_samples: 每个类别最多保留的样本数
        """
        self.bird_name_mapping = bird_name_mapping
        self.augment = augment
        self.img_size = img_size
        self.model_type = model_type
        self.input_channel = input_channel  # 新增控制通道参数
        self.use_xymasking = use_xymasking
        self.max_samples = max_samples

        # 读取数据
        df = pd.read_csv(csv_path)

        # ✅ 限制每个类别最多 `max_samples` 个样本
        self.data = self.limit_samples_per_class(df)

        # 定义数据增强（如果启用）
        self.transform = self.get_transforms() if augment else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据样本
        """
        row = self.data.iloc[idx]
        file_path = row["path"]
        bird_name = row["bird_name"]
        label = self.bird_name_mapping.get(bird_name, -1)

        # 读取频谱图并检查维度
        spectrogram = np.load(file_path).astype(np.float32)  # (H, W)

        # 检查 NaN 或 Inf
        if np.isnan(spectrogram).any() or np.isinf(spectrogram).any():
            print(f"❌ 数据错误: {file_path} 包含 NaN 或 Inf！")
            np.save("E:/AMR/DA/Projekt/data/error_sample.npy", spectrogram)  # 保存出错样本
            raise ValueError(f"数据 {file_path} 含有 NaN/Inf")      

        # 归一化到 [0, 1]
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

        # 确保 shape 为 (1, H, W)
        spectrogram = np.expand_dims(spectrogram, axis=0)  # (1, H, W)

        # 数据增强（如果启用）
        if self.augment and self.transform:
            spectrogram = self.apply_transforms(spectrogram)

        # **如果 input_channel = 3，则扩展到 3 通道**
        if self.input_channel == 3:
            spectrogram = np.repeat(spectrogram, 3, axis=0)  # (3, H, W)

        return torch.tensor(spectrogram), label, file_path

    def limit_samples_per_class(self, df):
        """
        限制每个类别最多 `max_samples` 个样本
        """
        balanced_data = []
        for bird_name in df["bird_name"].unique():
            class_data = df[df["bird_name"] == bird_name]
            if len(class_data) > self.max_samples:
                class_data = class_data.sample(n=self.max_samples, random_state=42)
            balanced_data.append(class_data)
        
        return pd.concat(balanced_data).reset_index(drop=True)

    def get_transforms(self):
        """
        返回数据增强策略
        """
        return albu.Compose([
            albu.HorizontalFlip(p=0.5),  # 水平翻转
            albu.XYMasking(
                p=0.3,
                num_masks_x=(1, 3),
                num_masks_y=(1, 3),
                mask_x_length=(1, 10),
                mask_y_length=(1, 20),
            ) if self.use_xymasking else albu.NoOp()
        ])

    def apply_transforms(self, spectrogram):
        """
        应用数据增强（Albumentations 需要 HWC 格式，因此转换格式）
        """
        spectrogram = spectrogram.squeeze(0)  # (H, W) 去掉通道维度
        spectrogram = np.expand_dims(spectrogram, axis=-1)  # (H, W, 1) 适配 Albumentations

        # 进行数据增强
        augmented = self.transform(image=spectrogram)["image"]

        # **确保最终形状 (1, H, W)**
        return augmented.transpose(2, 0, 1)  # (H, W, 1) → (1, H, W)



import numpy as np
import pandas as pd
import torch
import random
import cv2
from torch.utils.data import Dataset

import os
import torch
import numpy as np
import pandas as pd
import albumentations as albu
import cv2
import random
from torch.utils.data import Dataset

class BirdCLEFDataset(Dataset):
    def __init__(self, csv_path, bird_name_mapping, augment=False, img_size=(256, 256), model_type="EfficientNet",
                 input_channel=1, use_xymasking=True, max_samples=2500):
        """
        鸟类音频频谱数据集
        :param csv_path: 训练/测试数据的 CSV 文件路径
        :param bird_name_mapping: 字典，bird_name -> label 映射
        :param augment: 是否进行数据增强
        :param img_size: 频谱图目标尺寸
        :param model_type: 选择模型类型 ("EfficientNet", "EfficientNetV2", "ReXNet")
        :param input_channel: 选择输入通道数（1 = 单通道，3 = 复制为三通道）
        :param use_xymasking: 是否使用 XY 遮挡（已删除，不再使用）
        :param max_samples: 每个类别最多保留的样本数
        """
        self.bird_name_mapping = bird_name_mapping
        self.augment = augment
        self.img_size = img_size
        self.model_type = model_type
        self.input_channel = input_channel  # 新增控制通道参数
        self.use_xymasking = use_xymasking  # 不再使用，但保留参数避免改动代码结构
        self.max_samples = max_samples

        # 读取数据
        df = pd.read_csv(csv_path)
        df = df[df["bird_name"].isin(bird_name_mapping.keys())]
        # ✅ 限制每个类别最多 `max_samples` 个样本
        self.data = self.limit_samples_per_class(df)
        print(f"📊 数据集构建完成：共 {len(self.data)} 条样本，类别数：{len(df['bird_name'].unique())}")
        # 定义数据增强（如果启用）
        self.transform = self.get_transforms() if augment else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据样本
        """
        row = self.data.iloc[idx]
        file_path = row["mel_path"]
        bird_name = row["bird_name"]
        label = self.bird_name_mapping.get(bird_name, -1)

        # 读取频谱图并检查维度
        spectrogram = np.load(file_path).astype(np.float32)  # (H, W)

        # 检查 NaN 或 Inf
        if np.isnan(spectrogram).any() or np.isinf(spectrogram).any():
            print(f"❌ 数据错误: {file_path} 包含 NaN 或 Inf！")
            np.save("E:/AMR/DA/Projekt/data/error_sample.npy", spectrogram)  # 保存出错样本
            raise ValueError(f"数据 {file_path} 含有 NaN/Inf")      

        # 归一化到 [0, 1]
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

        # 确保 shape 为 (1, H, W)
        spectrogram = np.expand_dims(spectrogram, axis=0)  # (1, H, W)

        # 数据增强（如果启用）
        if self.augment:
            spectrogram = self.apply_transforms(spectrogram)

        # **如果 input_channel = 3，则扩展到 3 通道**
        if self.input_channel == 3:
            spectrogram = np.repeat(spectrogram, 3, axis=0)  # (3, H, W)

        return torch.tensor(spectrogram), label, file_path

    def limit_samples_per_class(self, df):
        """
        限制每个类别最多 `max_samples` 个样本。
        如果包含 `source` 列，则优先保留 small 数据集中所有样本。
        """
        balanced_data = []

        if "source" not in df.columns:
            # ✅ 原始逻辑：每类最多 max_samples
            for bird_name in df["bird_name"].unique():
                class_data = df[df["bird_name"] == bird_name]
                if len(class_data) > self.max_samples:
                    class_data = class_data.sample(n=self.max_samples, random_state=42)
                balanced_data.append(class_data)
            print("🧩 使用默认采样策略（无 source 字段）")
        else:
            # ✅ 新逻辑：small 全保留，big 限量采样
            for bird_name in df["bird_name"].unique():
                class_data = df[df["bird_name"] == bird_name]
                small_data = class_data[class_data["source"] == "small"]
                big_data = class_data[class_data["source"] == "big"]

                # 对 big 限制数量
                if len(big_data) > self.max_samples:
                    big_data = big_data.sample(n=self.max_samples, random_state=42)

                combined = pd.concat([small_data, big_data])
                balanced_data.append(combined)

            print("🧩 使用合并数据采样策略（包含 source 字段）")

        return pd.concat(balanced_data).reset_index(drop=True)


    def get_transforms(self):
        """
        返回新的数据增强策略（Vertical Roll + Warping）
        """
        return None  # 这里不再使用 Albumentations，而是改用自定义方法

    def apply_transforms(self, spectrogram):
        """
        应用新的数据增强（Vertical Roll + Warping + Gaussian Noise）
        """
        spectrogram = spectrogram.squeeze(0) # (H, W) 去掉通道维度

        # **1. Vertical Roll**
        if np.random.rand() < 0.5:
            spectrogram = self.vertical_roll(spectrogram)

        # **2. Warping**
        if np.random.rand() < 0.5:
            spectrogram = self.time_frequency_warping(spectrogram)

        # ✅ 3. Gaussian Noise（新增）
        if np.random.rand() < 0.5:
            spectrogram = self.add_gaussian_noise(spectrogram, std=0.01)

        # **确保最终形状 (1, H, W)**
        return np.expand_dims(spectrogram, axis=0)

    def vertical_roll(self, spectrogram, max_shift=20):
        """
        频率轴滚动（Vertical Roll），模拟频率偏移
        :param spectrogram: 输入声谱图 (H, W)
        :param max_shift: 最大偏移像素数
        """
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(spectrogram, shift, axis=0)  # 在频率轴上滚动
    
    def add_gaussian_noise(self, spectrogram, mean=0.0, std=0.01):
        """
        添加高斯噪声
        :param spectrogram: 输入图像 (H, W)
        :param mean: 高斯均值
        :param std: 标准差
        """
        noise = np.random.normal(mean, std, spectrogram.shape).astype(np.float32)
        noisy = spectrogram + noise
        return np.clip(noisy, 0, 1) # 保持在 [0, 1] 区间

    def time_frequency_warping(self, spectrogram, max_warp=10):
        """
        频谱扭曲（Warping），使用仿射变换实现
        :param spectrogram: 输入声谱图 (H, W)
        :param max_warp: 最大扭曲像素数
        """
        h, w = spectrogram.shape
        src = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        dst = np.array([
            [random.randint(-max_warp, max_warp), random.randint(-max_warp, max_warp)],
            [w + random.randint(-max_warp, max_warp), random.randint(-max_warp, max_warp)],
            [random.randint(-max_warp, max_warp), h + random.randint(-max_warp, max_warp)],
            [w + random.randint(-max_warp, max_warp), h + random.randint(-max_warp, max_warp)]
        ], dtype=np.float32)

        # 计算仿射变换矩阵
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped_spectrogram = cv2.warpPerspective(spectrogram, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        return warped_spectrogram




def get_dataloader(csv_path, bird_name_mapping, batch_size=32, num_workers=4, augment=False, use_weighted_sampler=True, model_type="EfficientNet", input_channel=3, max_samples=3000):
    """
    获取 DataLoader，支持均衡采样
    :param csv_path: CSV 文件路径
    :param bird_name_mapping: 训练时的类别映射
    :param batch_size: 批大小
    :param num_workers: DataLoader 进程数
    :param augment: 是否进行数据增强
    :param use_weighted_sampler: 是否使用 `WeightedRandomSampler`
    :param model_type: 选择模型类型 ("EfficientNet", "EfficientNetV2")
    :return: DataLoader
    """
    # ✅ 读取数据集
    dataset = BirdCLEFDataset(csv_path=csv_path, bird_name_mapping=bird_name_mapping, input_channel=input_channel, augment=augment, model_type=model_type, max_samples=max_samples)

    if use_weighted_sampler:
        # ✅ 计算类别权重
        labels = dataset.data["bird_name"].map(bird_name_mapping).values
        class_counts = np.bincount(labels, minlength=len(bird_name_mapping))  # 统计每个类别的数量
        class_weights = 1.0 / class_counts  # 计算权重（样本少的类别权重更高）
        weights = class_weights[labels]  # 为每个样本分配权重

        # ✅ 生成 `WeightedRandomSampler`
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        # ✅ 创建 DataLoader（均衡采样）
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    
    else:
        # ✅ 普通 DataLoader（不均衡采样）
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import random

if __name__ == "__main__":
    """
    测试 Dataset，打印数据增强前后的效果，并计算增强前后的差异指标
    """
    # 设定 sample_index（None = 随机选取）
    sample_index = None
    my_csvPath = "E:/AMR/DA/Projekt/data/train_list_for_zoom006_0314.csv"
    my_bird_mapping = {
        "Eurasian Blue tit": 0,
        "Eurasian Bullfinch": 1,
        "Great Tit": 2,
        "Hawfinch": 3,
        "Hooded Crow": 4,
        "Stock Dove": 5,
        "Background Noise": 6
    }

    # 创建 Dataset（训练集，带数据增强）
    dataset = BirdCLEFDataset(csv_path=my_csvPath, bird_name_mapping=my_bird_mapping, augment=True)

    # 随机选择一个样本
    if sample_index is None:
        sample_index = random.randint(0, len(dataset) - 1)

    # 读取数据（增强前）
    dataset.augment = False  # 先关闭数据增强
    original_spec, label, _ = dataset[sample_index]

    # 读取数据（增强后）
    dataset.augment = True  # 开启数据增强
    augmented_spec, _, _ = dataset[sample_index]
    
    # 打印形状确认一致
    print(f"Original shape: {original_spec.shape}, Augmented shape: {augmented_spec.shape}")

    # 将 Torch Tensor 转回 NumPy
    original_spec = original_spec.squeeze().numpy()  # (H, W)
    augmented_spec = augmented_spec.squeeze().numpy()  # (H, W)

    # 计算 MSE（均方误差）
    mse_value = np.mean((original_spec - augmented_spec) ** 2)

    # 计算 SSIM（结构相似性）
    ssim_value = ssim(original_spec, augmented_spec, data_range=augmented_spec.max() - augmented_spec.min())

    # 打印计算结果
    print(f"🔹 MSE (Mean Squared Error): {mse_value:.6f}")
    print(f"🔹 SSIM (Structural Similarity Index): {ssim_value:.6f}")

    # 绘制图像
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original_spec, cmap="magma")
    axes[0].set_title(f"Before augmentation (Label: {label})")

    axes[1].imshow(augmented_spec, cmap="magma")
    axes[1].set_title(f"After augmentation (Label: {label})")

    plt.show()
