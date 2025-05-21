import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import cv2

class BirdSoundLSTMDataset(Dataset):
    def __init__(self, csv_path, bird_name_mapping, augment=False, max_samples=3000):
        self.bird_name_mapping = bird_name_mapping
        self.augment = augment
        self.max_samples = max_samples

        df = pd.read_csv(csv_path)
        df = df[df["bird_name"].isin(bird_name_mapping.keys())]
        self.data = self.limit_samples_per_class(df)
        print(f"📊 LSTM 数据集构建完成：共 {len(self.data)} 条样本，类别数：{len(self.data['bird_name'].unique())}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = row["mel_path"]
        label = self.bird_name_mapping[row["bird_name"]]

        mel = np.load(file_path).astype(np.float32)  # shape: (256, 512)
        mel = mel.T  # → shape: (512, 256)

        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)

        if self.augment:
            mel = self.apply_transforms(mel)

        return torch.tensor(mel), label
        # return torch.tensor(mel, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


    def limit_samples_per_class(self, df):
        balanced_data = []
        for bird_name in df["bird_name"].unique():
            class_data = df[df["bird_name"] == bird_name]
            if len(class_data) > self.max_samples:
                class_data = class_data.sample(n=self.max_samples, random_state=42)
            balanced_data.append(class_data)
        return pd.concat(balanced_data).reset_index(drop=True)
    
    def apply_transforms(self, mel):
        # if np.random.rand() < 0.5:
        #     mel = self.time_mask(mel, max_width=30)
        # if np.random.rand() < 0.5:
        #     mel = self.freq_mask(mel, max_width=20)
        # if np.random.rand() < 0.5:
        #     mel = self.add_noise(mel, noise_level=0.01)
        if np.random.rand() < 0.5:
            mel = self.vertical_roll(mel, max_shift=20)
        if np.random.rand() < 0.5:
            mel = self.add_gaussian_noise(mel, std=0.01)
        if np.random.rand() < 0.5:
            mel = self.time_frequency_warping(mel, max_warp=10)
        return mel

    def time_mask(self, mel, max_width=30):
        t = mel.shape[0]
        mask_width = np.random.randint(1, max_width)
        start = np.random.randint(0, t - mask_width)
        mel[start:start+mask_width, :] = 0
        return mel

    def freq_mask(self, mel, max_width=20):
        f = mel.shape[1]
        mask_width = np.random.randint(1, max_width)
        start = np.random.randint(0, f - mask_width)
        mel[:, start:start+mask_width] = 0
        return mel

    def add_noise(self, mel, noise_level=0.01):
        noise = np.random.normal(0, noise_level, mel.shape).astype(np.float32)
        return mel + noise
    
    def vertical_roll(self, mel, max_shift=20):
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(mel, shift, axis=1)  # 注意：频率轴是第1维

    def add_gaussian_noise(self, mel, mean=0.0, std=0.01):
        noise = np.random.normal(mean, std, mel.shape).astype(np.float32)
        return np.clip(mel + noise, 0, 1)

    def time_frequency_warping(self, mel, max_warp=10):
        h, w = mel.shape
        src = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        dst = np.array([
            [random.randint(-max_warp, max_warp), random.randint(-max_warp, max_warp)],
            [w + random.randint(-max_warp, max_warp), random.randint(-max_warp, max_warp)],
            [random.randint(-max_warp, max_warp), h + random.randint(-max_warp, max_warp)],
            [w + random.randint(-max_warp, max_warp), h + random.randint(-max_warp, max_warp)],
        ], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(mel, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)



def get_loader(csv_path, bird_name_mapping, batch_size=32, num_workers=4, augment=False, use_weighted_sampler=True, max_samples=2000):
    dataset = BirdSoundLSTMDataset(
        csv_path=csv_path,
        bird_name_mapping=bird_name_mapping,
        augment=augment,
        max_samples=max_samples
    )

    if use_weighted_sampler:
        labels = dataset.data["bird_name"].map(bird_name_mapping).values
        class_counts = np.bincount(labels, minlength=len(bird_name_mapping))
        class_weights = 1.0 / class_counts
        weights = class_weights[labels]

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
