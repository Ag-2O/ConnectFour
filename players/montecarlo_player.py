from __future__ import annotations
""" モンテカルロ法で行動するプレイヤー """

"""
インポート
"""
import random
import copy as cp
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from game import Game
from players import Player, RandomPlayer
from dataclass.const import Const

class MonteCarloPlayer(Player):
    """
    モンテカルロプレイヤー
    """
    def __init__(self, turn: int, simulation_num: int = 100):
        super().__init__(turn)
        self._simulation_num = simulation_num

    def initialize(self):
        pass

    def act(self, game:Game) -> 'tuple[int,int]':
        """
        行動決定
        """
        # 合法手の取得
        legal_actions: list[tuple[int,int]] = game.get_leagl_actions()

        # ランダムに行動を取りあい勝率を測定
        win_rates: list[float] = []
        for act in legal_actions:
            win_num: int = 0
            for _ in range(self._simulation_num):
                # ゲームのコピー
                copy_game: Game = cp.deepcopy(game)

                # 対象の行動を実行
                copy_game.progress(act)

                # ランダムプレイヤーをセット
                copy_game.players = [RandomPlayer(Const.FIRST_PLAYER), RandomPlayer(Const.SECOND_PLAYER)]

                # 終局までプレイ
                while copy_game.status != Const.GAME_STATUS_FINISHED:
                    # 現在のプレイヤー
                    current = copy_game.players[game.cur_index]

                    # 行動決定
                    pos: 'tuple[int,int]' = current.act(copy_game)

                    # 実行
                    copy_game.progress(pos)
                
                # 勝ちをカウント
                if (copy_game.winner == self._turn):
                    win_num += 1
            
            # 勝率を計算して格納
            try:
                win_rate: float = float(win_num / self._simulation_num)
                win_rates.append(win_rate)
                #print(f"{act}: {win_rate}")
            except:
                win_rates.append(float(0))

        # 最大値のインデックスを取得
        max_idx: int = win_rates.index(max(win_rates))

        # 行動選択
        try:
            action: 'tuple[int,int]' = legal_actions[max_idx]
        except:
            action = legal_actions[0]

        return action

    def finalize(self):
        pass

    def __str__(self):
        return "MonteCarloPlayer"