import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn
import timm

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, model_type="EfficientNet", version="b0", pretrained=True, feature_extract=False, input_channel=3, dropout_rate=0.0):
        """
        统一的模型加载类，支持：
        - EfficientNetV1（b0-b7）
        - EfficientNetV2（s/m/l）
        - ReXNet（rexnet_100 / rexnet_200 / rexnet_300 / rexnet_350）
        
        :param num_classes: 分类类别数
        :param model_type: 选择模型类型 ("EfficientNet", "EfficientNetV2", "ReXNet")
        :param version: 选择 EfficientNet 版本 ("b0"-"b7") 或 EfficientNetV2 ("s", "m", "l") 或 ReXNet ("100", "200", "300", "350")
        :param pretrained: 是否加载 ImageNet 预训练参数
        :param feature_extract: 是否只训练分类层
        :param input_channel: 输入通道数（1=单通道灰度图, 3=RGB图像）
        """
        super(EfficientNetModel, self).__init__()

        # **选择模型**
        if model_type == "EfficientNet":
            model_name = f"efficientnet_{version}"  # e.g., efficientnet_b0
        elif model_type == "EfficientNetV2":
            model_name = f"efficientnetv2_{version}"  # e.g., efficientnetv2_s
        elif model_type == "ReXNet":
            model_name = f"rexnet_{version}"  # e.g., rexnet_300
        else:
            raise ValueError(f"无效的模型类型 '{model_type}'，请使用 'EfficientNet', 'EfficientNetV2', 'ReXNet'")

        # **使用 `timm` 加载模型**
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout_rate)

        # **如果 input_channel 不是 3，则修改第一层卷积层**
        if input_channel != 3:
            self.modify_input_channel(input_channel)
        print(f"✅ 分类层输出维度: {self.model.classifier.out_features}")
        # **是否冻结特征提取层**
        if feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False

    def modify_input_channel(self, input_channel):
        """
        修改模型的 `conv_stem` 层，使其适应自定义 `input_channel`（1 或 3）。
        """
        # 获取 `conv_stem` 原始的输出通道数
        conv_out_channels = self.model.conv_stem.out_channels
        
        # **修改第一层卷积，使其适应 input_channel**
        self.model.conv_stem = nn.Conv2d(input_channel, conv_out_channels, kernel_size=3, stride=2, padding=1, bias=False)

        # **修改 `bn1` 以适应新的 `conv_stem`**
        self.model.bn1 = nn.BatchNorm2d(conv_out_channels)

        print(f"✅ 修改 `conv_stem` 输入通道: {input_channel} → {conv_out_channels}")

    def forward(self, x):
        return self.model(x)



# import torch
# import torch.nn as nn
# import torchvision.models as models

# import torch
# import torch.nn as nn
# from torchvision import models

# import timm  # 使用 timm 加载 EfficientNetV2

# import torch
# import torch.nn as nn
# import timm
# from torchvision import models

# class EfficientNetModel(nn.Module):
#     def __init__(self, num_classes, model_type="EfficientNet", version="b0", pretrained=True, feature_extract=False):
#         """
#         EfficientNet 预训练模型，支持不同版本 (b0, b1, b2, b3, b4, b5, b6, b7) 和 EfficientNetV2
#         :param num_classes: 分类类别数
#         :param model_type: 选择模型类型 ("EfficientNet", "EfficientNetV2")
#         :param version: 选择 EfficientNet 版本 ("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7") 或 EfficientNetV2 ("s", "m", "l")
#         :param pretrained: 是否加载 ImageNet 预训练参数
#         :param feature_extract: 是否只训练分类层
#         """
#         super(EfficientNetModel, self).__init__()

#         # 判断选择 EfficientNet 还是 EfficientNetV2
#         if model_type == "EfficientNet":
#             effnet_versions = {
#                 "b0": models.efficientnet_b0,
#                 "b1": models.efficientnet_b1,
#                 "b2": models.efficientnet_b2,
#                 "b3": models.efficientnet_b3,
#                 "b4": models.efficientnet_b4,
#                 "b5": models.efficientnet_b5,
#                 "b6": models.efficientnet_b6,
#                 "b7": models.efficientnet_b7,
#             }
#             if version not in effnet_versions:
#                 raise ValueError(f"无效的 EfficientNet 版本 '{version}'，请使用 {list(effnet_versions.keys())}")
#             # 加载 EfficientNet 预训练模型
#             self.model = effnet_versions[version](pretrained=pretrained)

#             # 修改第一个卷积层，使其适配单通道输入 (C=1)
#             self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

#             # 修改分类头，使其适配 `num_classes`
#             num_ftrs = self.model.classifier[1].in_features
#             self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

#             # 修改 BatchNorm 层，确保它适应单通道输入
#             for layer in self.model.features:
#                 if isinstance(layer, nn.BatchNorm2d):
#                     layer.num_features = 32  # 假设你把输入的通道数设置为32

#         elif model_type == "EfficientNetV2":
#             # 使用 timm 加载 EfficientNetV2
#             self.model = timm.create_model(f"efficientnetv2_{version}", pretrained=pretrained, num_classes=num_classes)
#         else:
#             raise ValueError(f"无效的模型类型 '{model_type}'，请使用 'EfficientNet' 或 'EfficientNetV2'")

#         # 是否冻结特征提取层，仅训练分类头
#         if feature_extract:
#             for param in self.model.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         return self.model(x)

      

if __name__ == "__main__":
    """
    测试 EfficientNet 版本选择
    """
    num_classes = 3  # 假设有 3 个鸟类分类
    model_version = "b0"  # 可以改成 "b0", "b1", "b2", "b3"
    
    # 创建模型
    model = EfficientNetModel(num_classes, version=model_version, pretrained=False, feature_extract=True)

    # 随机生成输入数据 (batch_size=1, channels=1, height=256, width=256)
    test_input = torch.randn(1, 1, 256, 256)

    # 前向传播
    output = model(test_input)

    # 打印结果
    print(f"\n使用 EfficientNet-{model_version} 版本")
    print("模型结构:\n", model)
    print("\n输入张量形状:", test_input.shape)
    print("输出张量形状:", output.shape)  # 期望形状: [1, num_classes]
