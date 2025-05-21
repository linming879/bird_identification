import os
os.environ["ALBUMENTATIONS_DISABLE_CHECK"] = "1"
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from model.model import BirdSoundLSTM, BirdSoundCNNLSTM_Attention, BirdSoundLSTM_Attn, BirdSoundCNNLSTM_MultiHeadAttn  # ✅ 你需要提前放好
from data.dataset import BirdSoundLSTMDataset, get_loader  # ✅ 自定义 Dataset
from data.gen_mapping import generate_bird_name_mapping_with_tit_group

def train():
    # 鸟类标签（根据你的任务）
    bird_names = [
        "Black-headed Gull", "Canada Goose", "Carrion Crow", "Common Chaffinch",
        "Common Kingfisher", "Common Redstart", "Dunnock", "Eurasian Blackbird",
        "Eurasian Blackcap", "Eurasian Blue Tit", "Eurasian Bullfinch", "Eurasian Coot",
        "Eurasian Golden Oriole", "Eurasian Jay", "Eurasian Nuthatch", "Eurasian Siskin",
        "Eurasian Treecreeper", "Eurasian Wren", "European Goldfinch", "European Robin",
        "Goldcrest", "Great Spotted Woodpecker", "Great Tit", "Hawfinch", "Hooded Crow",
        "Long-tailed Tit", "Mallard", "Marsh Tit", "Redwing", "Rook", "Short-toed Treecreeper",
        "Stock Dove", "Background Noise"
    ]

    bird_name_mapping_path = 'E:/AMR/DA/Projekt/bird_cls_cnn/projekt_lstm/bird_name_mapping.py'
    bird_name_mapping = generate_bird_name_mapping_with_tit_group(
        bird_names=bird_names, output_path=bird_name_mapping_path,
        merge_tit_to_one=False, tit_class_id=0
    )

    exp_name = "exp_lstm_add_attn_mel_hidenLayer128_multihead_cnndataAug"
    datalist_root = "E:/AMR/DA/Projekt/data/data_list/0408"

    CONFIG = {
        "train_csv": f"{datalist_root}/train_list_high_quality.csv",
        "valid_csv": f"{datalist_root}/valid_list_high_quality.csv",
        "num_classes": len(set(bird_name_mapping.values())),
        "batch_size": 32,
        "num_epochs": 30,
        "valid_interval": 1,
        "check_interval": 1,
        "early_stop_patience": 3,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "resume_training": False,
        "use_scheduler": False,
        "checkpoint_dir": f"E:/AMR/DA/Projekt/bird_cls_cnn/projekt_LSTM/experiments/{exp_name}/models",
        "experiment_dir": f"E:/AMR/DA/Projekt/bird_cls_cnn/projekt_LSTM/experiments/{exp_name}",
        "log_dir": f"E:/AMR/DA/Projekt/bird_cls_cnn/projekt_LSTM/experiments/{exp_name}/logs",
        "hidden_dim": 128,
        "num_layers": 2,
        "bidirectional": True,
        "max_sample_num": 2000,
        "use_weighted_sampler": True,
        "num_workers": 4,
    }

    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0
        for x, y in tqdm(loader, desc="Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(loader)

    def validate_epoch(model, loader, criterion, device):
        model.eval()
        running_loss = 0
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                running_loss += loss.item()
        return running_loss / len(loader)

    os.makedirs(CONFIG["experiment_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    current_script = os.path.abspath(__file__)
    destination_path = os.path.join(CONFIG["experiment_dir"], "train_backup.py")
    shutil.copy(current_script, destination_path)

    train_loader = get_loader(
        csv_path=CONFIG["train_csv"],
        bird_name_mapping=bird_name_mapping,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        augment=True,
        use_weighted_sampler=CONFIG["use_weighted_sampler"],
        max_samples=CONFIG["max_sample_num"]
    )

    val_loader = get_loader(
        csv_path=CONFIG["valid_csv"],
        bird_name_mapping=bird_name_mapping,
        batch_size=CONFIG["batch_size"], 
        num_workers=CONFIG["num_workers"], 
        augment=False,
        use_weighted_sampler=None,
        max_samples=CONFIG["max_sample_num"]
        )

    # 模型选择 ===============================
    # model = BirdSoundLSTM(
    #     input_dim=256,
    #     hidden_dim=CONFIG["hidden_dim"],
    #     num_layers=CONFIG["num_layers"],
    #     num_classes=CONFIG["num_classes"],
    #     bidirectional=CONFIG["bidirectional"]
    # ).to(CONFIG["device"])
    
    # model = BirdSoundCNNLSTM_Attention(
    #     input_dim=256,
    #     hidden_dim=CONFIG["hidden_dim"],
    #     num_layers=CONFIG["num_layers"],
    #     num_classes=CONFIG["num_classes"],
    #     bidirectional=CONFIG["bidirectional"]
    # ).to(CONFIG["device"])

    # model = BirdSoundLSTM_Attn(
    #     input_dim=256,
    #     hidden_dim=CONFIG["hidden_dim"],
    #     num_layers=CONFIG["num_layers"],
    #     num_classes=CONFIG["num_classes"],
    #     bidirectional=CONFIG["bidirectional"]
    # ).to(CONFIG["device"])

    model = BirdSoundCNNLSTM_MultiHeadAttn(
        input_dim=256,
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        num_classes=CONFIG["num_classes"],
        bidirectional=CONFIG["bidirectional"],
        num_heads=4 # 可调参数
    ).to(CONFIG["device"])
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True) if CONFIG["use_scheduler"] else None

    writer = SummaryWriter(CONFIG["log_dir"])
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n📅 Epoch {epoch+1}/{CONFIG['num_epochs']}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        val_loss = validate_epoch(model, val_loader, criterion, CONFIG["device"])

        print(f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"))
            print("✅ 最优模型已保存！")
        else:
            early_stop_counter += 1
            print(f"⚠️ 验证未提升（{early_stop_counter}/{CONFIG['early_stop_patience']}）")
            if early_stop_counter >= CONFIG["early_stop_patience"]:
                print("🛑 提前停止训练！")
                break

        if (epoch + 1) % CONFIG["check_interval"] == 0:
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"📌 检查点已保存: {ckpt_path}")

    writer.close()
    print("🎉 LSTM 训练完成！")

if __name__ == "__main__":
    # 🚀 启动训练主程序
    train()

