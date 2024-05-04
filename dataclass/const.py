""" 定数 """

class Const:
    """
    定数クラス
        BOARD_WIDTH : 盤面の横のサイズ
        BOARD_HEIGHT : 盤面の縦のサイズ
        WIN_COLOR_LENGTH : 勝利条件
        GAME_STATUS_PROGRESSED: ゲーム進行中
        GAME_STATUS_FINISHED: ゲーム終了
        WINNER_DRAW: 引き分けorゲーム進行中
        FIRST_PLAYER: 先攻
        SECOND_PLAYER: 後攻
        SEARCH_DIRECTIONS: 探索の8方向
        REPROGRESS_LIMIT: 再実行可能回数
    """
    # 盤面の横のサイズ
    BOARD_WIDTH = 7

    # 盤面の縦のサイズ
    BOARD_HEIGHT = 6

    # 勝利条件
    WIN_COLOR_LENGTH = 4

    # ゲームの状態：進行中
    GAME_STATUS_PROGRESSED: int = 0

    # ゲームの状態：終了
    GAME_STATUS_FINISHED: int = 1

    # 引き分け
    WINNER_DRAW: int = 0

    # 先攻
    FIRST_PLAYER: int = 1

    # 後攻
    SECOND_PLAYER: int = -1

    # 探索方向
    SEARCH_DIRECTIONS: 'list[tuple[int,int]]' = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1)]

    # 再実行可能回数
    REPROGRESS_LIMIT = 2