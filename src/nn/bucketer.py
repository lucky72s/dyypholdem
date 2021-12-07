import math
import pickle

import torch

import settings.arguments as arguments
import settings.game_settings as game_settings

import nn.bucketing.river_tools as river_tools
import nn.bucketing.turn_tools as turn_tools
import nn.bucketing.flop_tools as flop_tools
import game.card_tools as card_tools


_initialized = False
_preflop_buckets: torch.Tensor
_flop_cats: dict
_turn_cats: dict
_ihr_pair_to_bucket: dict
_river_ihr: dict
_river_buckets = 1000


def initialize():
    global _initialized
    global _preflop_buckets
    global _flop_cats
    global _turn_cats
    global _ihr_pair_to_bucket
    global _river_ihr
    global _river_buckets

    if not _initialized:

        arguments.timer.start("Initializing buckets", log_level="TRACE")

        _preflop_buckets = None

        arguments.timer.split_start(message="Initializing river buckets", log_level="TRACE")
        _ihr_pair_to_bucket = {}
        _ihr_pair_to_bucket = pickle.load(open("./nn/bucketing/ihr_pair_to_bucket.pkl", "rb"))
        _river_buckets = len(_ihr_pair_to_bucket)
        arguments.timer.split_stop(message="River buckets initialized in", log_level="LOADING")

        arguments.timer.split_start(message="Initializing flop categories", log_level="TRACE")
        _flop_cats = pickle.load(open("./nn/bucketing/flop_dist_cats.pkl", "rb"))
        arguments.timer.split_stop(message="Flop categories initialized in", log_level="LOADING")

        if not arguments.use_sqlite:
            arguments.timer.split_start(message="Initializing turn categories", log_level="TRACE")
            _turn_cats = pickle.load(open("./nn/bucketing/turn_dist_cats.pkl", "rb"))
            arguments.timer.split_stop(message="Turn categories initialized in", log_level="LOADING")

            arguments.timer.split_start(message="Initializing river categories", log_level="TRACE")
            _river_ihr = pickle.load(open("./nn/bucketing/river_ihr.pkl", "rb"))
            arguments.timer.split_stop(message="River categories initialized in", log_level="LOADING")

        arguments.timer.stop(message="Bucket initialization done in", log_level="TIMING")

    _initialized = True


initialize()


# --- Gives the total number of buckets across all boards.
# -- @return the number of buckets
def get_bucket_count(street):
    if street == 4:
        return _river_buckets
    elif street == 3 or street == 2:
        return 1000
    elif street == 1:
        return 169
    return 169


# --- Gives the maximum number of ranks across all boards.
# -- @return the number of buckets
def get_rank_count():
    return math.comb(14, 2) + math.comb(10, 2)


# --- Gives a vector which maps private hands to buckets on a given board.
# -- @param board a non-empty vector of board cards
# -- @return a vector which maps each private hand to a bucket index
def compute_buckets(board):
    street = card_tools.board_to_street(board)
    if street == 1:
        return _compute_preflop_buckets()
    elif street == 2:
        return _compute_flop_buckets(board)

    if arguments.use_sqlite:
        if street == 3:
            return _compute_turn_buckets_sql(board)
        elif street == 4:
            return _compute_river_buckets_sql(board)
    else:
        if street == 3:
            return _compute_turn_buckets(board)
        elif street == 4:
            return _compute_river_buckets(board)


def compute_rank_buckets(board):
    raise NotImplementedError()


def _compute_preflop_buckets():
    global _preflop_buckets

    if _preflop_buckets is None:
        _preflop_buckets = arguments.Tensor(game_settings.hand_count).fill_(-1)

        for card1 in range(game_settings.card_count):
            for card2 in range(card1 + 1, game_settings.card_count):
                idx = card_tools.get_hole_index([card1, card2])
                rank1 = math.floor(card1 / 4)
                rank2 = math.floor(card2 / 4)
                if card1 % 4 == card2 % 4:
                    _preflop_buckets[idx] = rank1 * 13 + rank2 + 1
                else:
                    _preflop_buckets[idx] = rank2 * 13 + rank1 + 1
    return _preflop_buckets


def _compute_flop_buckets(board: torch.Tensor) -> torch.Tensor:
    buckets = arguments.Tensor(game_settings.hand_count).fill_(-1)
    _board = board.sort().values
    board_size = board.size(0)
    used = [0] * game_settings.card_count
    hand = [0] * (board_size + game_settings.hand_card_count)

    for i in range(0, board_size):
        used[int(_board[i].item())] = 1
        hand[i + game_settings.hand_card_count] = int(_board[i].item())

    for card1 in range(0, game_settings.card_count):
        if used[card1] == 0:
            used[card1] = 1
            hand[0] = card1
            for card2 in range(card1 + 1, game_settings.card_count):
                if used[card2] == 0:
                    used[card2] = 1
                    hand[1] = card2
                    idx = card_tools.get_hole_index([card1, card2])
                    flop_code = flop_tools.flop_id(hand.copy())
                    closest_mean = _flop_cats[flop_code]
                    if closest_mean is None:
                        raise IndexError()
                    buckets[idx] = closest_mean
                    used[card2] = 0
            used[card1] = 0
    return buckets


def _compute_turn_buckets(board) -> torch.Tensor:
    buckets = arguments.Tensor(game_settings.hand_count).fill_(-1)
    _board = board.sort().values
    board_size = board.size(0)
    used = [0] * game_settings.card_count
    hand = [0] * (board_size + game_settings.hand_card_count)

    for i in range(0, board_size):
        used[int(_board[i].item())] = 1
        hand[i + game_settings.hand_card_count] = int(_board[i].item())

    for card1 in range(0, game_settings.card_count):
        if used[card1] == 0:
            used[card1] = 1
            hand[0] = card1
            for card2 in range(card1 + 1, game_settings.card_count):
                if used[card2] == 0:
                    used[card2] = 1
                    hand[1] = card2
                    idx = card_tools.get_hole_index([card1, card2])
                    turn_code = turn_tools.turn_id(hand.copy())
                    closest_mean = _turn_cats[turn_code]
                    if closest_mean is None:
                        raise IndexError()
                    buckets[idx] = closest_mean
                    used[card2] = 0
            used[card1] = 0
    return buckets


def _compute_turn_buckets_sql(board) -> torch.Tensor:
    import sqlite3

    buckets = arguments.Tensor(game_settings.hand_count).fill_(-1)
    _board = board.sort().values
    board_size = board.size(0)
    used = [0] * game_settings.card_count
    hand = [0] * (board_size + game_settings.hand_card_count)

    for i in range(0, board_size):
        used[int(_board[i].item())] = 1
        hand[i + game_settings.hand_card_count] = int(_board[i].item())

    select_str = "select * from turn_cats where turn_id=:_id"
    with sqlite3.connect("./nn/bucketing/bucketing_data.sqlite") as conn:
        cur = conn.cursor()
        for card1 in range(0, game_settings.card_count):
            if used[card1] == 0:
                used[card1] = 1
                hand[0] = card1
                for card2 in range(card1 + 1, game_settings.card_count):
                    if used[card2] == 0:
                        used[card2] = 1
                        hand[1] = card2
                        idx = card_tools.get_hole_index([card1, card2])
                        turn_id = {"_id": turn_tools.turn_id(hand.copy())}
                        cur.execute(select_str, turn_id)
                        closest_mean = cur.fetchone()
                        if closest_mean is None:
                            raise IndexError()
                        buckets[idx] = closest_mean[1]
                        used[card2] = 0
                used[card1] = 0
    return buckets


def _compute_river_buckets(board) -> torch.Tensor:
    buckets = arguments.Tensor(game_settings.hand_count).fill_(-1)
    _board = board.sort().values
    board_size = board.size(0)
    used = [0] * game_settings.card_count
    hand = [0] * (board_size + game_settings.hand_card_count)

    for i in range(0, board_size):
        used[int(_board[i].item())] = 1
        hand[i + game_settings.hand_card_count] = int(_board[i].item())

    for card1 in range(0, game_settings.card_count):
        if used[card1] == 0:
            used[card1] = 1
            hand[0] = card1
            for card2 in range(card1 + 1, game_settings.card_count):
                if used[card2] == 0:
                    used[card2] = 1
                    hand[1] = card2
                    idx = card_tools.get_hole_index([card1, card2])
                    river_code = river_tools.river_id(hand)
                    ihr = _river_ihr[river_code]
                    if ihr is None:
                        raise IndexError()
                    else:
                        win_bucket = ihr[0]
                        tie_bucket = math.floor(ihr[1] / 2)
                    river_bucket = _ihr_pair_to_bucket[win_bucket * 1000 + tie_bucket]
                    assert river_bucket is not None, 'bad win, tie, ihr pair'
                    buckets[idx] = river_bucket
                    used[card2] = 0
            used[card1] = 0
    return buckets


def _compute_river_buckets_sql(board) -> torch.Tensor:
    import sqlite3

    buckets = arguments.Tensor(game_settings.hand_count).fill_(-1)
    _board = board.sort().values
    board_size = board.size(0)
    used = [0] * game_settings.card_count
    hand = [0] * (board_size + game_settings.hand_card_count)

    for i in range(0, board_size):
        used[int(_board[i].item())] = 1
        hand[i + game_settings.hand_card_count] = int(_board[i].item())

    select_str = "select * from river_ihr where river_id=:_id"
    with sqlite3.connect("./nn/bucketing/bucketing_data.sqlite") as conn:
        cur = conn.cursor()
        for card1 in range(0, game_settings.card_count):
            if used[card1] == 0:
                used[card1] = 1
                hand[0] = card1
                for card2 in range(card1 + 1, game_settings.card_count):
                    if used[card2] == 0:
                        used[card2] = 1
                        hand[1] = card2
                        idx = card_tools.get_hole_index([card1, card2])
                        river_id = {"_id": river_tools.river_id(hand)}
                        cur.execute(select_str, river_id)
                        ihr = cur.fetchone()
                        if ihr is None:
                            raise IndexError()
                        else:
                            win_bucket = ihr[1]
                            tie_bucket = math.floor(ihr[2] / 2)
                        river_bucket = _ihr_pair_to_bucket[win_bucket * 1000 + tie_bucket]
                        assert river_bucket is not None, 'bad win, tie, ihr pair'
                        buckets[idx] = river_bucket
                        used[card2] = 0
                used[card1] = 0
    return buckets

