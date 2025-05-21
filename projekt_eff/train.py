import os
os.environ["ALBUMENTATIONS_DISABLE_CHECK"] = "1"
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from data.dataset import get_dataloader
from data.gen_mapping import generate_bird_name_mapping_with_tit_group
from model.model import EfficientNetModel
from tqdm import tqdm
import numpy as np
import importlib.util

# === 配置鸟类映射 ===
bird_names = [
    "Black-headed Gull", "Canada Goose", "Carrion Crow", "Common Chaffinch", "Common Kingfisher",
    "Common Redstart", "Dunnock", "Eurasian Blackbird", "Eurasian Blackcap", "Eurasian Blue Tit",
    "Eurasian Bullfinch", "Eurasian Coot", "Eurasian Golden Oriole", "Eurasian Jay", "Eurasian Nuthatch",
    "Eurasian Siskin", "Eurasian Treecreeper", "Eurasian Wren", "European Goldfinch", "European Robin",
    "Goldcrest", "Great Spotted Woodpecker", "Great Tit", "Hawfinch", "Hooded Crow", "Long-tailed Tit",
    "Mallard", "Marsh Tit", "Redwing", "Rook", "Short-toed Treecreeper", "Stock Dove", "Background Noise"
]

bird_name_mapping_path = 'E:/AMR/DA/Projekt/bird_cls_cnn/projekt_eff/bird_name_mapping.py'
def load_mapping_from_path(py_path):
    spec = importlib.util.spec_from_file_location("bird_name_mapping", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bird_name_mapping

if not os.path.exists(bird_name_mapping_path):
    bird_name_mapping = generate_bird_name_mapping_with_tit_group(
        bird_names=bird_names,
         output_path=bird_name_mapping_path,
        merge_tit_to_one=False,
        tit_class_id=0
    )
else:
    bird_name_mapping = load_mapping_from_path(bird_name_mapping_path)


# === 配置参数 ===
datalist_root = 'E:/AMR/DA/Projekt/data/data_list/0408'
exp_name = 'exp11_alltype0408_mel_drop0.2_addgauss_b1'

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
    "model_type": "EfficientNet",
    "model_version": "b1",
    "input_channel": 1,
    "pretrained": True,
    "feature_extract": False,
    "resume_training": False,
    "flag_weighted_sample": True,
    "checkpoint_dir": f"E:/AMR/DA/Projekt/bird_cls_cnn/projekt_eff/experiments/{exp_name}/models",
    "experiment_dir": f"E:/AMR/DA/Projekt/bird_cls_cnn/projekt_eff/experiments/{exp_name}",
    "dropout_rate": 0.2,
    "max_samples": 2000,
    "use_scheduler": False
}

def train(model, CONFIG):
    writer = SummaryWriter(os.path.join(CONFIG["experiment_dir"], "logs"))

    train_loader = get_dataloader(CONFIG["train_csv"], bird_name_mapping, batch_size=CONFIG["batch_size"],
                                  augment=True, use_weighted_sampler=CONFIG["flag_weighted_sample"],
                                  model_type=CONFIG["model_type"], num_workers=4,
                                  input_channel=CONFIG["input_channel"], max_samples=CONFIG["max_samples"])
    test_loader = get_dataloader(CONFIG["valid_csv"], bird_name_mapping, batch_size=CONFIG["batch_size"],
                                 augment=False, use_weighted_sampler=False,
                                 model_type=CONFIG["model_type"], num_workers=4,
                                 input_channel=CONFIG["input_channel"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True) if CONFIG["use_scheduler"] else None

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        writer.add_scalar("Loss/Train", train_loss, epoch)

        if (epoch + 1) % CONFIG["valid_interval"] == 0 or epoch == CONFIG["num_epochs"] - 1:
            val_loss = validate_epoch(model, test_loader, criterion, CONFIG["device"])
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            print(f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

            if scheduler:
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"⚠️ 验证损失未提升 ({early_stop_counter}/{CONFIG['early_stop_patience']})")
                if early_stop_counter >= CONFIG["early_stop_patience"]:
                    print("🛑 早停触发，结束训练！")
                    break

            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            print(f"✅ 新的最优模型已保存: {checkpoint_path}")

        if (epoch + 1) % CONFIG["check_interval"] == 0:
            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            print(f"📌 检查点已保存: {checkpoint_path}")

    writer.close()
    print("🎉 训练完成！")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels, paths in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels, paths in tqdm(test_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(test_loader)

if __name__ == "__main__":
    print("🧠 PyTorch 版本:", torch.__version__)
    print("🖥️ CUDA 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("✅ 使用的 GPU:", torch.cuda.get_device_name(0))
        print("🔢 当前设备 ID:", torch.cuda.current_device())
    else:
        print("⚠️ 未检测到可用的 GPU")
    print("cuDNN 可用:", torch.backends.cudnn.is_available())

    os.makedirs(CONFIG["experiment_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    current_script = os.path.abspath(__file__)
    destination_path = os.path.join(CONFIG["experiment_dir"], "train_backup.py")
    shutil.copy(current_script, destination_path)

    model = EfficientNetModel(CONFIG["num_classes"], model_type=CONFIG["model_type"], version=CONFIG["model_version"],
                              pretrained=CONFIG["pretrained"], feature_extract=CONFIG["feature_extract"],
                              input_channel=CONFIG["input_channel"], dropout_rate=CONFIG["dropout_rate"])
    model.to(CONFIG["device"])

    train(model, CONFIG)
