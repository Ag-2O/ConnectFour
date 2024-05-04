from __future__ import annotations
""" 盤面と行動履歴のクラスのコード """

"""
インポート
"""
from dataclass.const import Const

class GameHistory:
    """
    盤面と行動履歴のクラス
    """
    def __init__(self, board: 'list[list[int]]', pos: 'tuple[int,int]'):
        self.board: 'list[list[int]]' = board
        self.pos_1d: int = trans_pos_2d_to_1d(pos)
        self.winner: int = 0
    
    def set_winner(self, winner: int) -> None:
        """
        勝者をセットする
        """
        self.winner = winner

def trans_pos_2d_to_1d(pos:'tuple[int,int]') -> int:
    """
    2次元座標から1次元座標へ変換する
    """
    y: int = pos[1] * Const.BOARD_WIDTH
    x: int = pos[0]

    return y + x