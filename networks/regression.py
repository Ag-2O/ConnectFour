""" ネットワークモデルコード """

"""
インポート
"""
import numpy as np
import torch
from util import diagonal_to_horizontal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RegressionModel(nn.Module):
    """
    盤面情報から勝率を推定するネットワーク
    """
    def __init__(self):
        """
        コンストラクタ
        """
        super().__init__()

        # 畳み込み1層目（入力チャンネル: 1, 出力チャンネル: 4）
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1
            )

        # 畳み込み2層目（入力チャンネル: 4, 出力チャンネル: 16）
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=3,
            stride=1
            )

        # 畳み込み3層目（入力チャンネル: 16, 出力チャンネル: 64）
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=64,
            kernel_size=2,
            stride=1
            )

        # バッチ正則化
        self.bn1 = nn.BatchNorm2d(num_features=4)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.bn3 = nn.BatchNorm2d(num_features=64)

        # 平滑化
        self.flt = nn.Flatten()

        # 全結合層（入力: チャンネル数×特徴マップのサイズ, 出力: 1）
        self.fc = nn.Linear(
            in_features=64 * 1 * 2,
            out_features=1
            )

    def forward(self, x):
        """
        勝率の推論（順伝播関数）, __call__メソッドがforward()を呼び出す
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flt(x)
        x = self.fc(x)
        x = F.tanh(x)
        return x

class MultipleCNNRegressionModel(nn.Module):
    """
    盤面情報から勝率を推定するネットワーク（改）
    """
    def __init__(self):
        """
        コンストラクタ
        """
        super().__init__()

        # 縦方向の畳み込み 1
        self.height_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(4,1),
            stride=1
        )

        # 縦方向の畳み込み 2
        self.height_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3,1),
            stride=1
        )

        # 横方向の畳み込み 1
        self.width_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(1,4),
            stride=1
        )

        # 横方向の畳み込み 2
        self.width_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(1,3),
            stride=1
        )

        # 左下から右上への斜め方向の畳み込み 1
        self.diagonal_up_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(1,4),
            stride=1
        )

        # 左下から右上への斜め方向への畳み込み 2
        self.diagonal_up_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(1,3),
            stride=1
        )

        # 左上から右下への斜め方向の畳み込み 1
        self.diagonal_down_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(1,4),
            stride=1
        )

        # 左上から右下への斜め方向への畳み込み 2
        self.diagonal_down_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(1,3),
            stride=1
        )

        # 平滑化
        self.flatten = nn.Flatten()

        # バッチ正則化
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        # 全結合層（入力: チャンネル数×特徴マップのサイズ, 出力: 1）
        self.fc = nn.Linear(
            in_features=64 * 1 * 4,
            out_features=1
            )

    def forward(self, x):
        """
        勝率の推論（順伝播関数）, __call__メソッドがforward()を呼び出す
        """

        # 縦方向
        h = F.relu(self.bn1(self.height_conv1(x)))
        h = F.relu(self.bn2(self.height_conv2(h)))
        h = F.avg_pool2d(h, h.size()[2:]).view(h.size(0), -1)
        #print(f"h: {h.size()}")

        # 横方向
        w = F.relu(self.bn1(self.width_conv1(x)))
        w = F.relu(self.bn2(self.width_conv2(w)))
        w = F.avg_pool2d(w, w.size()[2:]).view(w.size(0), -1)
        #print(f"w: {w.size()}")

        # 斜め方向（左上から右下）
        dd = diagonal_to_horizontal(x)
        dd = F.relu(self.bn1(self.diagonal_up_conv1(dd)))
        dd = F.relu(self.bn2(self.diagonal_up_conv2(dd)))
        dd = F.avg_pool2d(dd, dd.size()[2:]).view(dd.size(0), -1)
        #print(f"dd: {dd.size()}")

        # 斜め方向（左下から右上）
        du = diagonal_to_horizontal(x, True)
        du = F.relu(self.bn1(self.diagonal_down_conv1(du)))
        du = F.relu(self.bn2(self.diagonal_down_conv2(du)))
        du = F.avg_pool2d(du, du.size()[2:]).view(du.size(0), -1)
        #print(f"du: {du.size()}")

        x = torch.cat([h,w,dd,du], dim=1)
        x = F.tanh(self.fc(x))
        return x


class FCRegressionModel(nn.Module):
    """
    盤面情報から勝率を推定するネットワーク(FCのみ)
    """
    def __init__(self):
        """
        コンストラクタ
        """
        super().__init__()

        # 全結合層1層目
        self.fc1 = nn.Linear(
            in_features=42,
            out_features=256
            )

        # 全結合層2層目
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=256
            )

        # 全結合層3層目
        self.fc3 = nn.Linear(
            in_features=256,
            out_features=1
            )

    def forward(self, x):
        """
        勝率の推論（順伝播関数）, __call__メソッドがforward()を呼び出す
        """
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x