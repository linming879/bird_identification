import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def visualize_gradcam(sample_path, true_label, pred_label, model, index_to_bird, device="cuda"):

    # ✅ 加载 mel 图
    spec = np.load(sample_path)

    # ✅ 原始 mel 是低频在上，这里翻转使高频在下
    spec = np.flipud(spec)

    # ✅ 预处理：防止 NaN 和黑线（核心改进）
    spec = np.nan_to_num(spec, nan=0.0)

    # 若最大值和最小值相差太小，直接跳过归一化，避免除以 0
    if np.max(spec) - np.min(spec) < 1e-6:
        spec = np.zeros_like(spec)
    else:
        spec = np.interp(spec, (spec.min(), spec.max()), (0, 1)) # 替代归一化

    # ✅ 转为输入
    input_tensor = torch.tensor(spec).unsqueeze(0).unsqueeze(0).float().to(device)

    # ✅ 获取类别 index
    pred_index = list(index_to_bird.keys())[list(index_to_bird.values()).index(pred_label)]
    targets = [ClassifierOutputTarget(pred_index)]

    # ✅ GradCAM 初始化
    target_layer = model.model.blocks[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    cam.activations_and_grads.release()

    # ✅ RGB 图
    spec_rgb = np.stack([spec] * 3, axis=-1)
    visualization = show_cam_on_image(spec_rgb, grayscale_cam, use_rgb=True)

    # ✅ 显示（只设置 origin='upper'，保持高频在下）
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(spec, aspect="auto", origin="upper", cmap="viridis")
    plt.title("🎼 Original Mel-Spectrogram (High Freq ↓)")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"🔥 Grad-CAM - True: {true_label}, Pred: {pred_label}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
