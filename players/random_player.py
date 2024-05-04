from __future__ import annotations
""" ランダムに行動するプレイヤー """

"""
インポート
"""
import random
from players import Player
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from game import Game

class RandomPlayer(Player):
    """
    ランダムプレイヤー
    """
    def __init__(self, turn: int):
        super().__init__(turn)

    def initialize(self):
        pass

    def act(self, game:Game) -> 'tuple[int,int]':
        """
        行動決定
        """
        # 合法手の取得
        legal_actions: list[tuple[int,int]] = game.get_leagl_actions()

        # ランダムに選択
        action: tuple[int,int] = random.choice(legal_actions)

        return action

    def finalize(self):
        pass

    def __str__(self):
        return "RandomPlayer"