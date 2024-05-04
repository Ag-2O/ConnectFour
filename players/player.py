from __future__ import annotations
""" プレイヤーの抽象クラス """

"""
インポート
"""
from abc import ABCMeta
from abc import abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from game import Game

class Player(metaclass = ABCMeta):
    """
    プレイヤーの抽象クラス
    """

    # コンストラクタ
    @abstractmethod
    def __init__(self, turn: int):
        self._turn = turn

    # 初期化
    @abstractmethod
    def initialize(self):
        pass

    # 行動
    @abstractmethod
    def act(self, game:Game) -> 'tuple[int, int]':
        pass

    # 終了処理
    @abstractmethod
    def finalize(self):
        pass

    # 名前出力
    @abstractmethod
    def __str__(self):
        return "Player"