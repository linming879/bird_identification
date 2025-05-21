
import os
import json
import librosa
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from model.model import build_model
from bird_name_mapping import bird_name_mapping


index_to_bird = {v: k for k, v in bird_name_mapping.items()}


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def audio_to_melspec(audio_data, cfg):
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, sr=cfg["fs"], n_fft=cfg["n_fft"],
        hop_length=cfg["n_fft"] - cfg["win_lap"],
        win_length=cfg["win_size"], n_mels=cfg["spec_size"][1],
        fmin=cfg["min_freq"], fmax=cfg["max_freq"], window='hann'
    )
    mel_spec = np.log10(mel_spec + 1e-9)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
    mel_spec = cv2.resize(mel_spec, cfg["spec_size"], interpolation=cv2.INTER_AREA)
    return mel_spec


class InferenceDataset(Dataset):
    def __init__(self, mels):
        self.mels = mels

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        mel = self.mels[idx]
        mel = mel.T  # (512, 256)
        return torch.tensor(mel, dtype=torch.float32)


def load_model(cfg):
    model_config = {
        "input_dim": cfg.get("input_dim", 256),
        "hidden_dim": cfg.get("hidden_dim", 128),
        "num_layers": cfg.get("num_layers", 2),
        "num_classes": cfg["num_classes"],
        "bidirectional": cfg.get("bidirectional", True),
        "dropout_rate": cfg.get("dropout_rate", 0.3),
        "num_heads": cfg.get("num_heads", 4),
    }

    model = build_model(cfg["gru_type"], model_config)
    state_dict = torch.load(cfg["model_path"], map_location=cfg["device"])
    model.load_state_dict(state_dict if isinstance(state_dict, dict) else state_dict["model_state_dict"])
    model.to(cfg["device"])
    model.eval()
    return model


def infer_audio(file_path, cfg):
    model = load_model(cfg)
    infer_output_dir = os.path.join(cfg["output_dir"], cfg["infer_name"])
    os.makedirs(infer_output_dir, exist_ok=True)

    audio_data, _ = librosa.load(file_path, sr=cfg["fs"])
    total_duration = len(audio_data) / cfg["fs"]
    num_segments = int(np.ceil(total_duration / cfg["segment_duration"]))

    segments, specs = [], []
    for i in range(num_segments):
        start = i * cfg["segment_duration"]
        end = (i + 1) * cfg["segment_duration"]
        seg_audio = audio_data[int(start * cfg["fs"]):int(end * cfg["fs"])]

        if len(seg_audio) < cfg["segment_duration"] * cfg["fs"]:
            seg_audio = np.pad(seg_audio, (0, int(cfg["segment_duration"] * cfg["fs"]) - len(seg_audio)))

        spec = audio_to_melspec(seg_audio, cfg)
        specs.append(spec)
        segments.append((format_time(start) + "-" + format_time(end), f"{start}-{end}", spec))

        img_dir = os.path.join(infer_output_dir, "infer_img")
        os.makedirs(img_dir, exist_ok=True)
        plt.imsave(os.path.join(img_dir, f"segment_{i:03d}.png"), spec, cmap="viridis", origin="lower")
        np.save(os.path.join(img_dir, f"segment_{i:03d}_mel.npy"), spec.astype(np.float32))

    dataset = InferenceDataset([s[2] for s in segments])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(cfg["device"])
            logits = model(batch)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    seg_results = [(segments[i][0], segments[i][1], index_to_bird[preds[i]]) for i in range(len(preds))]
    df_seg = pd.DataFrame(seg_results, columns=["Time Range", "Interval", "Predicted Bird"])
    df_seg.to_csv(os.path.join(infer_output_dir, "segments.csv"), index=False, encoding="utf-8-sig")

    merged = []
    last_label, start_time = None, None
    for tr, interval, label in seg_results:
        if label != last_label:
            if last_label is not None:
                merged.append((start_time, tr.split("-")[1], last_label))
            start_time = tr.split("-")[0]
            last_label = label
    if last_label:
        merged.append((start_time, tr.split("-")[1], last_label))
    df_merged = pd.DataFrame(merged, columns=["Start Time", "End Time", "Predicted Bird"])
    df_merged.to_csv(os.path.join(infer_output_dir, "merged_segments.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(infer_output_dir, "bird_counts.json"), "w", encoding="utf-8") as f:
        json.dump(df_merged["Predicted Bird"].value_counts().to_dict(), f, ensure_ascii=False, indent=4)
    with open(os.path.join(infer_output_dir, "bird_counts_raw.json"), "w", encoding="utf-8") as f:
        json.dump(df_seg["Predicted Bird"].value_counts().to_dict(), f, ensure_ascii=False, indent=4)

    print(f"✅ 推理完成，结果已保存至 {infer_output_dir}")
