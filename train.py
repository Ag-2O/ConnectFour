from __future__ import annotations
""" ゲームの学習コード """

"""
インポート
"""
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import util
from game import Game
from dataclass import Const, GameHistory, BoardDataset
from players import Player, RandomPlayer, MonteCarloPlayer
from networks import RegressionModel, MultipleCNNRegressionModel, FCRegressionModel

# 乱数の固定
torch.manual_seed(0)

def create_dataset(first: Player, second: Player, iter_num: int = 1000) -> None:
    """
    データセットを作成して保存する
    """    
    # データの作成
    histories: list[GameHistory] = util.iterate_game(iter_num, first, second)

    # データセットの作成
    dataset: Dataset = util.create_dataset(histories)

    # データセットの保存
    file_name: str = str(first) + "_vs_" + str(second) + "_" + str(iter_num)
    util.save_dataset(dataset, file_name)


def train_regression_model(num_epoch: int = 20) -> None:
    """
    回帰モデルの学習
    """
    # データセットの読み込み
    dataset: Dataset = util.load_dataset("20240503230336_MonteCarloPlayer_vs_RandomPlayer_2000_dataset.pickle")

    # データローダーの作成
    dataloaders: dict[DataLoader] = util.split_dataloader(dataset)

    # モデルのインスタンス作成
    #model: nn.Module = RegressionModel()
    model: nn.Module = MultipleCNNRegressionModel()
    #model: nn.Module = FCRegressionModel()

    # 損失関数の設定(最小二乗誤差)
    criterion: nn.MSELoss = nn.MSELoss()

    # 最適化手法の選択
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=0.01)

    # 損失関数の値のリスト
    l: list[np.ndarray] = []

    # エポック数だけ反復
    for epoch in tqdm(range(num_epoch)):
        # 学習モード
        model.train()
                
        epoch_loss = 0.0
        
        # DataLoaderを使ってデータを読み込む
        for batch in dataloaders["train"]:
            inputs = batch["board"]
            labels = batch["winner"]

            # 勾配を初期化する
            optimizer.zero_grad()
            
            # 推論
            outputs = model(inputs)

            # 損失関数を使って損失を計算する
            loss = criterion(outputs, labels)
            
            # 誤差を逆伝搬する
            loss.backward()

            # パラメータを更新する
            optimizer.step()
                
            epoch_loss += loss.item() * inputs.size(0)
                
        # 1エポックでの損失を計算
        epoch_loss = epoch_loss / len(dataloaders["train"].dataset)
        
        #lossをデータで保存する
        a_loss = np.array(epoch_loss)
        l.append(a_loss) 

        #epoch数とlossを表示する
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))   
        print('epoch_loss:{:.4f}'.format(epoch_loss))
        print('-'*20) 
        
        # カレントディレクトリの取得
        current_directory: str = os.getcwd()

        # パスの作成
        path: str = os.path.join(current_directory, f"models/best_regression_model.pth")
        
        #モデルを保存
        torch.save(model, path)

    # lossの出力
    print(l)

    pass


def train_classification_model() -> None:
    """
    分類モデルの学習
    """
    pass


def train_complex_model() -> None:
    """
    複合モデルの学習
    """
    pass

if __name__ == "__main__":
    #train_regression_model(100)
    #first: Player = MonteCarloPlayer(Const.FIRST_PLAYER, 100)
    #second: Player = RandomPlayer(Const.SECOND_PLAYER)
    #create_dataset(first, second, 2000)
    train_regression_model(2000)