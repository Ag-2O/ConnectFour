from __future__ import annotations
""" コネクトフォー本体 """

"""
インポート
"""
import numpy as np
import copy as cp
from players import Player
from dataclass import GameHistory, Const

class Game:
    """
    ゲームクラス
    """

    def __init__(self, first: Player, second: Player) -> None:
        """
        コンストラクタ
        """

        # 盤面
        self.board : list[list[int]] = [[0 for i in range(Const.BOARD_WIDTH)] for j in range(Const.BOARD_HEIGHT)]

        # プレイヤーの格納
        self.players: list[Player] = [first, second]

        # ゲームの状態
        self.status: int = Const.GAME_STATUS_PROGRESSED

        # ゲームの勝者
        self.winner: int = Const.WINNER_DRAW

        # プレイヤーカラー
        self._player_color: list[int] = [Const.FIRST_PLAYER, Const.SECOND_PLAYER]

        # 現在のプレイヤーのインデックス
        self.cur_index: int = 0

        # 盤面と行動の履歴
        self._histories: list[GameHistory] = []

        # 重力落下あり
        self.is_gravity = True
    

    def progress(self, pos: 'tuple[int,int]') -> bool:
        """
        ゲームの進行
        """
        # ゲームの状態が進行中の場合、ゲームを進める
        if (self.status == Const.GAME_STATUS_PROGRESSED):
            # 行動に失敗した場合、Falseを返す
            if (self._check_pos(pos) == False):
                return False
                
            # ゲームの盤面を更新
            if (self._update_board(pos) == False):
                return False
            
            # ゲームの勝者を判定する
            self._judge()

            # 次のプレイヤーへ
            self.cur_index = 0 if (self.cur_index == 1) else 1

            return True
        
        else:
            return False


    def _check_pos(self, pos:'tuple[int,int]') -> bool:
        """
        座標の判定
        """
        # 型チェック
        if (type(pos) != tuple):
            return False

        # 座標チェック
        pos_x: int = pos[0]
        pos_y: int = pos[1]
        if (pos_x < 0 or Const.BOARD_WIDTH <= pos_x or pos_y < 0 or Const.BOARD_HEIGHT <= pos_y):
            return False
        
        # 重複チェック
        pos_val: int = self.board[pos_y][pos_x]
        if (pos_val != 0):
            return False
        
        # 問題無し
        return True

    def _judge(self) -> None:
        """
        ゲームの判定
        """
        # ゲームの状態が進行中の場合
        if (self.status == Const.GAME_STATUS_PROGRESSED):
            # 自分の色のマスまで網羅
            cnt_without_zero:int = 0
            for y in range(Const.BOARD_HEIGHT):
                for x in range(Const.BOARD_WIDTH):
                    # 勝利条件を満たす場合、現在のプレイヤーの勝利とする
                    if (self._check_line((x,y))):
                        self.winner = self._player_color[self.cur_index]
                        self.status = Const.GAME_STATUS_FINISHED
                    else:
                        # 石が置かれていないマスを数える
                        if (self.board[y][x] == 0):
                            cnt_without_zero += 1
            
            # 空いているマスが0である場合
            if (self.status == Const.GAME_STATUS_PROGRESSED and cnt_without_zero == 0):
                # 引き分けとしてゲームを終了状態にする
                self.winner = Const.WINNER_DRAW
                self.status = Const.GAME_STATUS_FINISHED

        # ゲームが終了状態である場合、Trueを返す
        if (self.status == Const.GAME_STATUS_FINISHED):
            return True
        else:
            return False


    def _check_line(self, cur_pos: 'tuple[int,int]') -> bool:
        """
        連続したラインの確認
        """
        # 現在のプレイヤー情報
        cur_x: int = cur_pos[0]
        cur_y: int = cur_pos[1]
        cur_col: int = self._player_color[self.cur_index]
            
        # 与えられたマス目が現在のプレイヤーの色である場合
        if (self.board[cur_y][cur_x] == cur_col):
            # 全ての方向を探索する
            for dir in Const.SEARCH_DIRECTIONS:
                # 連続しているか確認
                cur_x = cur_pos[0]
                cur_y = cur_pos[1]
                cnt: int = 1
                while True:
                    cur_x = cur_x + dir[1]
                    cur_y = cur_y + dir[0]

                    # 盤面サイズを超える場合、終了
                    if (cur_x < 0 or Const.BOARD_WIDTH <= cur_x or cur_y < 0 or Const.BOARD_HEIGHT <= cur_y):
                        break

                    # 次のマス目の色が異なる場合、終了
                    elif (self.board[cur_y][cur_x] != cur_col):
                        break

                    # マスのカウント
                    else:
                        cnt += 1

                # 勝利条件のマス数以上である場合、Trueを返す
                if (Const.WIN_COLOR_LENGTH <= cnt):
                    return True

        # 勝利条件を満たしていない場合、Falseを返す
        return False


    def _update_board(self, pos: 'tuple[int,int]') -> bool:
        """
        重力を考慮して盤面を更新する
        """
        # 現在のプレイヤーの色
        color: int = self._player_color[self.cur_index]

        # 重力あり
        if (self.is_gravity):
            # 下方向に探索
            for y in range(Const.BOARD_HEIGHT):
                under_pos: int = y + 1

                # 最下段の場合、そのまま置く
                if (Const.BOARD_HEIGHT <= under_pos):
                    self.board[y][pos[0]] = color
                    return True
                
                # 下のマスに置かれている場合
                if (self.board[under_pos][pos[0]] != 0):
                    # そのまま置く
                    self.board[y][pos[0]] = color
                    return True
            
            # 置けなかった場合
            return False

        # 重力なし
        else:
            self.board[pos[1]][pos[0]] = color
            return True


    def get_leagl_actions(self) -> 'list[tuple[int,int]]':
        """
        合法手の取得
        """
        # まだ置かれてない位置を走査
        legal_actions: list[tuple[int,int]] = []
        for y in range(Const.BOARD_HEIGHT):
            for x in range(Const.BOARD_WIDTH):

                # 未だ置かれていない位置の場合
                if (self.board[y][x] == 0):

                    # 重力落下あり
                    if (self.is_gravity):
                        under_pos: int = y + 1

                        # 最下段なら可能
                        if (Const.BOARD_HEIGHT <= under_pos):
                            legal_actions.append((x,y))

                        # 下のマスが置かれている場合なら可能
                        else:
                            if (self.board[under_pos][x] != 0):
                                legal_actions.append((x,y))

                    # 重力落下なし
                    else:
                        legal_actions.append((x,y))
        
        return legal_actions


    def display_board(self) -> None:
        """
        盤面の標準出力
        """
        print("\n")

        for _ in range(Const.BOARD_WIDTH):
            print("--",end="")
        print("")

        for y in range(Const.BOARD_HEIGHT):
            print("|",end="")
            for x in range(Const.BOARD_WIDTH):
                if (self.board[y][x] == 1):
                    print("o|",end="")
                elif (self.board[y][x] == -1):
                    print("x|",end="")
                else:
                    print(" |",end="")
            print("")
        
        for _ in range(Const.BOARD_WIDTH):
            print("--",end="")
        print("")

    def set_game_history(self, pos: 'tuple[int,int]') -> None:
        """
        盤面と行動の履歴を保存する
        """