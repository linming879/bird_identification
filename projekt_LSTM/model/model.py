import torch
import torch.nn as nn
import torch.nn.functional as F

class BirdSoundLSTM(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=2, num_classes=3, bidirectional=True):
        super(BirdSoundLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        return self.classifier(pooled)


class BirdSoundLSTM_Attn(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=2, num_classes=3, bidirectional=True):
        super(BirdSoundLSTM_Attn, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 🌟 Attention 层
        self.attn = nn.Linear(lstm_output_dim, 1)

        # 分类器
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (B, T, D)
        attn_score = self.attn(lstm_out)  # (B, T, 1)
        attn_weights = F.softmax(attn_score, dim=1)  # (B, T, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, D)
        return self.classifier(context)


class BirdSoundCNNLSTM_Attention(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=2, num_classes=3, bidirectional=True):
        super(BirdSoundCNNLSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        #  1. CNN前处理：提取局部时频特征
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (256 → 128)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))   # (128 → 64)
        )

        #  得到 shape: (batch, 32, T/4=128, F/4=64)
        self.rnn_input_dim = 64 * 32  # 展平频率和通道作为特征

        #  2. LSTM 编码时序
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0.0
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        #  3. 注意力机制
        self.attention = nn.Linear(lstm_output_dim, 1)

        #  4. 分类器
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, time=512, freq=256) → reshape for CNN
        x = x.unsqueeze(1)  # (B, 1, 512, 256)

        cnn_feat = self.cnn(x)  # → (B, 32, 128, 64)
        B, C, T, F = cnn_feat.shape
        cnn_feat = cnn_feat.permute(0, 2, 1, 3).contiguous()  # → (B, T, C, F)
        rnn_input = cnn_feat.view(B, T, -1)  # → (B, T, 32*64)

        lstm_out, _ = self.lstm(rnn_input)  # (B, T, D)

        #  Attention 权重
        attn_scores = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # 时间维度归一化

        #  加权平均
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, D)

        return self.classifier(context)


import torch
import torch.nn as nn
import torch.nn.functional as F

class BirdSoundCNNLSTM_MultiHeadAttn(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=2, num_classes=3, bidirectional=True, num_heads=4):
        super(BirdSoundCNNLSTM_MultiHeadAttn, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_heads = num_heads

        # CNN 提取局部时频特征
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # CNN 输出特征展开后作为 LSTM 输入
        self.rnn_input_dim = 64 * 32  # C * F after pooling

        self.lstm = nn.LSTM(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0.0
        )

        self.lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 替换为多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.lstm_output_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.classifier = nn.Linear(self.lstm_output_dim, num_classes)

    def forward(self, x):
        # x: (B, T=512, F=256) → (B, 1, 512, 256)
        x = x.unsqueeze(1)

        # CNN → (B, 32, 128, 64)
        cnn_feat = self.cnn(x)
        B, C, T, F = cnn_feat.shape

        # reshape → (B, T, 2048)
        cnn_feat = cnn_feat.permute(0, 2, 1, 3).contiguous()
        rnn_input = cnn_feat.view(B, T, -1)

        lstm_out, _ = self.lstm(rnn_input)  # (B, T, D)

        # 多头注意力（Q=K=V=LSTM输出）
        attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)  # (B, T, D)

        # 时间维度均值池化
        context = attn_output.mean(dim=1)  # (B, D)

        return self.classifier(context)



class BirdSoundGRU_Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        output, _ = self.gru(x)
        out = output[:, -1, :]
        return self.fc(out)

class BirdSoundGRU_AvgPool(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        output, _ = self.gru(x)
        out = torch.mean(output, dim=1)
        return self.fc(out)

class BirdSoundGRU_Attn(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.attn = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        output, _ = self.gru(x)
        attn_weights = torch.softmax(self.attn(output), dim=1)
        out = torch.sum(attn_weights * output, dim=1)
        return self.fc(out)


class BirdSoundCNNGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=True):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # (256, 512) → (128, 256)
        )
        self.gru = nn.GRU(16 * 128, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.cnn(x)  # (B, 16, T//2, F//2)
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), x.size(2), -1)  # (B, T', C*F')
        output, _ = self.gru(x)
        out = output[:, -1, :]
        return self.fc(out)


class BirdSoundCNNGRU_Attn(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=True):
        super().__init__()
        # CNN block
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))  # 输入 (B, 1, 512, 256) → 输出 (B, 16, 256, 128)
        )

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # GRU 输入维度 = CNN 输出的通道数 * 压缩后的频率轴
        self.gru = nn.GRU(
            input_size=16 * 128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # 单头 Attention
        self.attn = nn.Linear(hidden_dim * self.num_directions, 1)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 512, 256)
        x = self.cnn(x)     # (B, 16, 256, 128)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), -1)  # (B, 256, 2048)
        output, _ = self.gru(x)  # (B, T, H)

        # attention
        attn_weights = torch.softmax(self.attn(output), dim=1)  # (B, T, 1)
        out = torch.sum(attn_weights * output, dim=1)           # (B, H)
        return self.fc(out)


class BirdSoundCNNGRU_MultiHeadAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=True, num_heads=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))  # (B, 1, 512, 256) → (B, 16, 256, 128)
        )

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=16 * 128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * self.num_directions,
            num_heads=num_heads,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 512, 256)
        x = self.cnn(x)     # (B, 16, 256, 128)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), -1)  # (B, 256, 2048)
        output, _ = self.gru(x)  # (B, T, H)

        attn_out, _ = self.attn(output, output, output)  # self-attention
        out = attn_out.mean(dim=1)  # 平均池化
        return self.fc(out)



class BirdSoundCNNGRU_Attn_Dropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=True, dropout_rate=0.3):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # CNN block + dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(dropout_rate)  # 🧱 CNN dropout
        )

        # GRU + dropout
        self.gru = nn.GRU(
            input_size=16 * 128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0  # PyTorch only uses dropout if num_layers > 1
        )

        self.attn = nn.Linear(hidden_dim * self.num_directions, 1)

        # Dropout before FC
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 512, 256)
        x = self.cnn(x)     # (B, 16, 256, 128)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), -1)  # (B, 256, 2048)

        output, _ = self.gru(x)  # (B, T, H)

        attn_weights = torch.softmax(self.attn(output), dim=1)  # (B, T, 1)
        out = torch.sum(attn_weights * output, dim=1)  # (B, H)

        out = self.dropout_fc(out)  # Dropout before classification
        return self.fc(out)



class BirdSoundCNNGRU_Attn_Dropout_ResNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=True, dropout_rate=0.3):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # CNN block
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(dropout_rate)
        )

        # GRU
        self.gru_input_dim = 16 * 128
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )

        # Residual 投影层：GRU输入 → GRU输出维度
        self.res_proj = nn.Linear(self.gru_input_dim, hidden_dim * self.num_directions)

        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)

        # Attention + FC
        self.attn = nn.Linear(hidden_dim * self.num_directions, 1)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)             # (B, 1, 512, 256)
        x = self.cnn(x)                # (B, 16, 256, 128)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), -1)  # (B, 256, 2048)

        gru_out, _ = self.gru(x)       # (B, T, H)

        # Residual + LayerNorm
        x_proj = self.res_proj(x)      # (B, T, H)
        gru_out = self.layer_norm(gru_out + x_proj)  # 残差加 LayerNorm

        # Attention
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)  # (B, T, 1)
        out = torch.sum(attn_weights * gru_out, dim=1)           # (B, H)

        out = self.dropout_fc(out)
        return self.fc(out)


    
def build_model(gru_type, config):
    if gru_type == "gru_base":
        return BirdSoundGRU_Baseline(config["input_dim"], config["hidden_dim"], config["num_layers"], config["num_classes"], config["bidirectional"])
    elif gru_type == "gru_avg":
        return BirdSoundGRU_AvgPool(config["input_dim"], config["hidden_dim"], config["num_layers"], config["num_classes"], config["bidirectional"])
    elif gru_type == "gru_attn":
        return BirdSoundGRU_Attn(config["input_dim"], config["hidden_dim"], config["num_layers"], config["num_classes"], config["bidirectional"])
    elif gru_type == "cnn_gru":
        return BirdSoundCNNGRU(config["input_dim"], config["hidden_dim"], config["num_layers"], config["num_classes"], config["bidirectional"])
    elif gru_type == "cnn_gru_attn":
        return BirdSoundCNNGRU_Attn(config["input_dim"], config["hidden_dim"], config["num_layers"], config["num_classes"], config["bidirectional"])
    elif gru_type == "cnn_gru_multihead":
        return BirdSoundCNNGRU_MultiHeadAttn(config["input_dim"], config["hidden_dim"], config["num_layers"], config["num_classes"], config["bidirectional"], num_heads=config["num_heads"])
    elif gru_type == "cnn_gru_attn_dropout":
        return BirdSoundCNNGRU_Attn_Dropout(config["input_dim"], config["hidden_dim"], config["num_layers"], config["num_classes"], config["bidirectional"], dropout_rate=config["dropout_rate"])
    elif gru_type == "cnn_gru_attn_resnorm":
        return BirdSoundCNNGRU_Attn_Dropout_ResNorm(config["input_dim"], config["hidden_dim"], config["num_layers"], config["num_classes"], config["bidirectional"], dropout_rate=config["dropout_rate"])
    else:
        raise ValueError(f"❌ Unknown gru_type: {gru_type}")
