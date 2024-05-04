from __future__ import annotations
""" ゲームの実行コード """

"""
インポート
"""
import numpy as np
import util
from game import Game
from dataclass import Const
from players import Player, RandomPlayer, MonteCarloPlayer

def play_one_game(first: Player, second: Player) -> int:
    """
    1ゲームの実行
    """
    # ゲームの定義
    game: Game = Game(first, second)

    # ゲームの進行
    while game.status != Const.GAME_STATUS_FINISHED:
        # 現在のプレイヤー
        current = game.players[game.cur_index]

        # 行動決定
        pos: 'tuple[int,int]' = current.act(game)

        # 実行
        is_progress: bool = game.progress(pos)
        print(f"is progress: {is_progress}")

        # 実行失敗時は0を返す
        if (is_progress == False):
            return 0

        # 盤面の出力
        game.display_board()
    
    # 勝利したプレイヤーを返す
    return game.winner


if __name__ == "__main__":
    # プレイヤーの定義
    first: Player = MonteCarloPlayer(Const.FIRST_PLAYER, 100)
    second: Player = RandomPlayer()

    # ゲーム実行
    winner: int = play_one_game(first, second)
    print(f"winner: {winner}")

    # 反復
    #util.iterate_game(10,first,second)
