import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import IncrementalPCA
import joblib
from tqdm import tqdm

class SVMSpectrogramDataset:
    def __init__(self, csv_path, label_mapping=None, max_samples_per_class=None,
                 pca_components=256, pca_batch_size=256,
                 pca_save_path=None, use_saved_pca=False):
        """
        支持 Incremental PCA 自动保存/加载 的数据集
        """
        self.df = pd.read_csv(csv_path)
        self.max_samples_per_class = max_samples_per_class
        self.pca_components = pca_components
        self.pca_batch_size = pca_batch_size
        self.pca_save_path = pca_save_path
        self.use_saved_pca = use_saved_pca

        if max_samples_per_class:
            self.df = self._limit_samples_per_class(self.df)

        self.label_encoder = LabelEncoder()
        if label_mapping:
            self.label_encoder.classes_ = np.array(list(label_mapping.keys()))
        else:
            self.label_encoder.fit(self.df["bird_name"])

        self.y = self.label_encoder.transform(self.df["bird_name"])
        self.bird_name_mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))

        self.X = self._incremental_pca()

    def _limit_samples_per_class(self, df):
        balanced_data = []
        for bird_name in df["bird_name"].unique():
            class_data = df[df["bird_name"] == bird_name]
            if len(class_data) > self.max_samples_per_class:
                class_data = class_data.sample(n=self.max_samples_per_class, random_state=42)
            balanced_data.append(class_data)
        return pd.concat(balanced_data).reset_index(drop=True)

    def _load_batch(self, batch_df):
        batch_data = []
        for _, row in batch_df.iterrows():
            path = row["mel_path"]
            try:
                spec = np.load(path).astype(np.float32)
                spec = (spec - spec.min()) / (spec.max() - spec.min())
                spec = spec.flatten()
                batch_data.append(spec)
            except Exception as e:
                print(f"❌ 读取失败: {path}, 错误: {e}")
        return np.array(batch_data)

    def _incremental_pca(self):
        n_samples = len(self.df)

        # 先尝试加载
        if self.use_saved_pca and self.pca_save_path and os.path.exists(self.pca_save_path):
            print(f"📂 加载已有 PCA 模型: {self.pca_save_path}")
            ipca = joblib.load(self.pca_save_path)
        else:
            print(f"🔄 开始 Incremental PCA 拟合，总样本数: {n_samples}")
            ipca = IncrementalPCA(n_components=self.pca_components, batch_size=self.pca_batch_size)

            # 第一轮 partial_fit
            for start_idx in tqdm(range(0, n_samples, self.pca_batch_size), desc="📦 拟合 PCA"):
                end_idx = min(start_idx + self.pca_batch_size, n_samples)
                batch_df = self.df.iloc[start_idx:end_idx]
                X_batch = self._load_batch(batch_df)
                if len(X_batch) > 0:
                    ipca.partial_fit(X_batch)

            # 保存拟合好的 PCA
            if self.pca_save_path:
                os.makedirs(os.path.dirname(self.pca_save_path), exist_ok=True)
                joblib.dump(ipca, self.pca_save_path)
                print(f"✅ PCA 模型已保存至: {self.pca_save_path}")

        # 第二轮 transform
        print("✨ 开始 Transform 数据...")
        X_all = []
        for start_idx in tqdm(range(0, n_samples, self.pca_batch_size), desc="📦 变换数据"):
            end_idx = min(start_idx + self.pca_batch_size, n_samples)
            batch_df = self.df.iloc[start_idx:end_idx]
            X_batch = self._load_batch(batch_df)
            if len(X_batch) > 0:
                X_transformed = ipca.transform(X_batch)
                X_all.append(X_transformed)

        return np.vstack(X_all)

    def get_data(self):
        return self.X, self.y

    def get_label_mapping(self):
        return self.bird_name_mapping
