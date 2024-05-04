from __future__ import annotations
""" いろいろ """

"""
インポート
"""
import os
import random
import copy as cp
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from dataclass import Const, GameHistory, BoardDataset
from game import Game
from players import Player

def trans_pos_2d_to_1d(pos:'tuple[int,int]') -> int:
    """
    2次元座標から1次元座標へ変換する
    """
    y: int = pos[1] * Const.BOARD_WIDTH
    x: int = pos[0]

    return y + x


def trans_pos_1d_to_2d(pos_1d: int) -> 'tuple[int,int]':
    """
    1次元座標から2次元座標へ変換する
    """
    y: int = pos_1d // Const.BOARD_WIDTH
    x: int = pos_1d % Const.BOARD_WIDTH
    return (x, y)


def trans_pos_horizontal_symmetry(pos_1d: int) -> int:
    """
    水平方向の対称変換（座標）
    """
    pos: 'tuple[int,int]' = trans_pos_1d_to_2d(pos_1d)
    pos = ((Const.BOARD_WIDTH - 1) - pos[0], pos[1])

    return trans_pos_2d_to_1d(pos)


def trans_board_horizontal_symmetry(board: 'list[list[int]]') -> 'list[list[int]]':
    """
    水平方向の対称変換（盤面）
    """
    transformed_board: list[list[int]] = []

    # 新しく行を作成して格納していく
    for row in board:
        transformed_row: list[int] = []

        # 逆順に追加して行を作成
        for elm in row[::-1]:
            transformed_row.append(elm)

        # 行の追加
        transformed_board.append(transformed_row)

    return transformed_board


def iterate_game(num: int, first: Player, second: Player) -> 'list[GameHistory]':
    """
    複数回プレイさせて履歴を返す
    """
    # 全ゲームの履歴リスト
    all_histories: list[GameHistory] = []

    # 実行したゲーム数 / 先攻 / 後攻 / 引分 / エラー
    rates: list[int] = [0, 0, 0, 0, 0]

    for n in tqdm(range(num)):
        # ゲームの定義
        if (random.random() < 0.5):
            game: Game = Game(first, second)
            first._turn = Const.FIRST_PLAYER
            second._turn = Const.SECOND_PLAYER
        else:
            game: Game = Game(second, first)
            first._turn = Const.SECOND_PLAYER
            second._turn = Const.FIRST_PLAYER

        # 1ゲームの履歴
        local_histories: list[GameHistory] = []

        # ゲームの進行
        while game.status != Const.GAME_STATUS_FINISHED:
            # 現在のプレイヤー
            current = game.players[game.cur_index]

            # 行動決定
            pos: 'tuple[int,int]' = current.act(game)

            # 実行
            if (game.progress(pos) == False):
                break

            # 履歴に格納
            local_histories.append(GameHistory(cp.deepcopy(game.board), pos))
        
        # 履歴に勝者を追加して、全履歴へ追加
        if (game.status == Const.GAME_STATUS_FINISHED):

            # 倍率の設定
            rate: int = 1.0
            for hist in local_histories[::-1]:
                hist.set_winner(game.winner)
                all_histories.append(hist)
                rate = rate * 0.99
            
            # 勝数のカウント
            rates[0] += 1
            if (game.winner == 1):
                rates[1] += 1
            elif (game.winner == -1):
                rates[2] += 1
            else:
                rates[3] += 1
        else:
            rates[4] += 1
    
    # 結果の出力
    print(f">> show results. \"finished game: {rates[0]}, first_win: {rates[1]}, second_win: {rates[2]}, draw: {rates[3]}, error: {rates[4]}\"")
    print(f">> history size: {len(all_histories)}")

    # 全履歴を返す
    return all_histories


def create_dataset(histories: list[GameHistory]) -> Dataset:
    """
    データセットを作成する
    """
    boards: 'list[list[list[int]]]' = []
    labels: 'list[tuple[int,int]]' = []
    
    # データの取り出し
    for hist in histories:
        # そのままデータを追加
        boards.append(hist.board)
        labels.append((hist.winner, hist.pos_1d))

        # 水平方向の変換
        transformed_board: list[list[int]] = trans_board_horizontal_symmetry(hist.board)
        transformed_pos_1d: int = trans_pos_horizontal_symmetry(hist.pos_1d)
    
        # 変換後のデータを追加
        boards.append(transformed_board)
        labels.append((hist.winner, transformed_pos_1d))

    # データセットの作成
    dataset: BoardDataset = BoardDataset(boards, labels)
    print(f">> dataset size: {len(dataset)}")

    return dataset


def save_dataset(dataset: Dataset, file_name: str) -> bool:
    """
    データセットをpickle化して保存する
    """
    try:
        # カレントディレクトリの取得
        current_directory: str = os.getcwd()

        # 現在の日付の取得
        now: datetime = datetime.now()
        now_str: str = now.strftime("%Y%m%d%H%M%S") + "_" + \
                       file_name + "_dataset.pickle"

        # パスの作成
        path: str = os.path.join(current_directory, f"datasets/{now_str}")

        # データセットをpickle化して保存
        with open(path, "wb") as f:
            pickle.dump(dataset, f)

        return True
    except Exception as e:
        print(e)

        return False


def load_dataset(file_name: str) -> Dataset:
    """
    pickle化されたデータセットを読み込む
    """
    loaded_dataset: Dataset = None
    try:
        # カレントディレクトリの取得
        current_directory: str = os.getcwd()

        # パスの作成
        path: str = os.path.join(current_directory, f"datasets/{file_name}")

        # pickleファイルを読み込む
        with open(path, "rb") as f:
            loaded_dataset = pickle.load(f)
        
    except Exception as e:
        print(e)
    
    return loaded_dataset


def split_dataloader(dataset: Dataset, train_rate: float = 0.6, batch_size: int = 32) -> 'dict[str, DataLoader]':
    """
    データローダーを分割して作成する
    """

    # 各データセットのサンプル数を決定（train : val : test = 60% : 20% : 20%）
    n_train: int = int(len(dataset) * train_rate)
    n_val: int = int((len(dataset) - n_train) * 0.5)
    n_test: int = len(dataset) - n_train - n_val

    # データセットの分割
    split_datasets: list[Dataset] = random_split(dataset, [n_train, n_val, n_test])

    # 各データローダーを作成
    train_loader = DataLoader(split_datasets[0], batch_size, shuffle=True)
    val_loader = DataLoader(split_datasets[1], batch_size)
    test_loader = DataLoader(split_datasets[2], batch_size)

    # データローダーをまとめる
    dataloaders: dict[str, DataLoader] = {"train": train_loader, "val": val_loader, "test": test_loader}

    return dataloaders


def diagonal_to_horizontal(tensor_board: torch.Tensor, is_reverse: bool = False) -> torch.Tensor:
    """
    盤面情報の対角成分を取り出して行成分に変換し、新しい盤面情報の行列を作成する
    """
    
    # 元のテンソルのサイズを取得
    size = tensor_board.size()

    # 斜めの成分を行に変換するための空のリストを作成
    rows = []

    # テンソル内の各斜めの成分を取り出す
    if (is_reverse):
        # 90度回転させる
        rotated_tensor_board = torch.rot90(tensor_board, k=1, dims=(-2, -1))

        for i in range(-(Const.BOARD_WIDTH-1), Const.BOARD_HEIGHT, 1):
            # 対角成分の取り出し（左下から右上）
            diagonal = torch.diagonal(rotated_tensor_board, offset=i, dim1=-2, dim2=-1)

            # 足りない部分を0でパディング
            padded: torch.Tensor = F.pad(diagonal, (0, Const.BOARD_WIDTH - diagonal.size()[-1]))

            # 次元を追加して行に変換し、リストに追加
            rows.append(padded.unsqueeze(dim=-2))

    else:
        for i in range(-(Const.BOARD_HEIGHT-1), Const.BOARD_WIDTH, 1):
            # 対角成分の取り出し（左下から右上）
            diagonal = torch.diagonal(tensor_board, offset=i, dim1=-2, dim2=-1)

            # 足りない部分を0でパディング
            padded: torch.Tensor = F.pad(diagonal, (0, Const.BOARD_HEIGHT - diagonal.size()[-1]))

            # 次元を追加して行に変換し、リストに追加
            rows.append(padded.unsqueeze(dim=-2))

    # 新しいテンソルを作成し、2次元を基にして結合
    sorted_tensor = torch.cat(rows, dim=-2)

    return sorted_tensor


if __name__ == "__main__":
    tensor = torch.randn((32,1,6,7))
    print(tensor[0][0])

    diagonal = diagonal_to_horizontal(tensor)
    print(diagonal[0][0])

    """
    pos: tuple[int,int] = (3,3) # 40

    history: GameHistory = GameHistory(board, pos)
    history.set_winner(Const.FIRST_PLAYER)

    create_dataset([history])
    """