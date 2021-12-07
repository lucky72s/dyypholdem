import math
from typing import List
from math import comb

import torch

import settings.arguments as arguments
import settings.constants as constants
import settings.game_settings as game_settings

import game.card_to_string_conversion as card_to_string

_flop_board_idx: torch.Tensor = None


# --- Gives whether a set of cards is valid.
# -- @param hand a vector of cards
# -- @return `true` if the tensor contains valid cards and no card is repeated
def hand_is_possible(hand):
    assert hand.min() >= 0 and hand.max() < game_settings.card_count, 'Illegal cards in hand'
    used_cards = arguments.Tensor(game_settings.card_count).fill_(0)
    for i in range(0, hand.size(0)):
        used_cards[hand[i]] += 1
    return used_cards.max() < 2


# --- Checks if a range vector is valid with a given board.
# -- @param range a range vector to check
# -- @param board a possibly empty vector of board cards
# -- @return `true` if the range puts 0 probability on invalid hands and has
# -- total probability 1
def is_valid_range(range, board):
    only_possible_hands = range.clone().mul_(get_impossible_hand_indexes(board)).sum() == 0
    sums_to_one = abs(1.0 - range.sum()) < 0.0001
    return only_possible_hands and sums_to_one


def get_possible_hands_mask(hands):
    used_cards = arguments.Tensor(hands.size(0), game_settings.card_count).fill_(0).long()
    used_cards.scatter_add_(1, hands, arguments.Tensor(hands.size(0), 7).fill_(1).long())
    ret = torch.le(torch.amax(used_cards, 1), 1)
    ret = ret.long()
    return ret


# --- Gives the private hands which are valid with a given board.
# -- @param board a possibly empty vector of board cards
# -- @return a vector with an entry for every possible hand (private card), which
# --  is `1` if the hand shares no cards with the board and `0` otherwise
def get_possible_hand_indexes(board):
    out = arguments.Tensor(game_settings.hand_count).fill_(0)
    if board.dim() == 0:
        out.fill_(1)
        return out

    used = [0] * game_settings.card_count
    if board.dim() == 1:
        for i in range(0, board.size(0)):
            used[board[i]] = 1

    for card1 in range(0, game_settings.card_count):
        if not used[card1]:
            for card2 in range(card1 + 1, game_settings.card_count):
                if not used[card2]:
                    out[get_hole_index([card1, card2])] = 1
    return out


# --- Gives the private hands which are invalid with a given board.
# -- @param board a possibly empty vector of board cards
# -- @return a vector with an entry for every possible hand (private card), which
# -- is `1` if the hand shares at least one card with the board and `0` otherwise
def get_impossible_hand_indexes(board):
    out = get_possible_hand_indexes(board)
    out.add_(-1)
    out.mul_(-1)
    return out


# -- Gives a numerical index for a set of hole cards.
# -- @param hand a non-empty vector of hole cards, sorted
# -- @return the numerical index for the hand
def get_hole_index(hand: List) -> int:
    index = 0
    for i in range(1, len(hand) + 1):
        index = index + comb(hand[i - 1], i)
    return index


# --- Gives a numerical index for a set of board cards.
# -- @param board a non-empty vector of board cards
# -- @return the numerical index for the board
def get_flop_board_index(board):
    global _flop_board_idx

    if _flop_board_idx is None:
        get_next_round_boards(arguments.Tensor())
    return _flop_board_idx[board[0]][board[1]][board[2]]


# --- Gives a numerical index for a set of hole cards.
# -- @param hand a non-empty vector of hole cards, sorted
# -- @return the numerical index for the hand
def string_to_hole_index(hand_string):
    hole = card_to_string.string_to_board(hand_string)
    hole, _ = hole.sort()
    index = 0
    for i in range(hole.size(0)):
        index = index + math.comb(hole[i], i+1)
    return index


# --- Gives the current betting round based on a board vector.
# -- @param board a possibly empty vector of board cards
# -- @return the current betting round
def board_to_street(board: torch.Tensor):
    if board.dim() == 0:
        return 1
    else:
        for i in range(0, constants.streets_count):
            if board.size(0) == game_settings.board_card_count[i]:
                return i + 1
    assert False, 'bad board dims'


# -- Gives all possible sets of board cards for the game.
# -- @return an NxK tensor, where N is the number of possible boards, and K is
# -- the number of cards on each board
def get_next_round_boards(board: torch.Tensor):
    global _flop_board_idx

    board_index = 0
    street = board_to_street(board)
    boards_count = get_next_boards_count(street)
    out = arguments.Tensor(boards_count, game_settings.board_card_count[street]).fill_(0.0)
    boards = arguments.Tensor(boards_count, game_settings.board_card_count[street]).fill_(0.0)
    cur_board = arguments.Tensor(game_settings.board_card_count[street]).fill_(0.0)
    if board.dim() > 0:
        for i in range(board.size(0)):
            cur_board[i] = board[i]

    _, out = _build_boards(boards, board_index, cur_board, out,
                           game_settings.board_card_count[street - 1] + 1,
                           game_settings.board_card_count[street],
                           game_settings.board_card_count[street - 1] + 1)

    if _flop_board_idx is None and board.nelement() == 0:
        _flop_board_idx = arguments.Tensor(game_settings.card_count, game_settings.card_count, game_settings.card_count).zero_().long()
        for i in range(boards_count):
            card1 = int(out[i][0].item())
            card2 = int(out[i][1].item())
            card3 = int(out[i][2].item())
            _flop_board_idx[card1][card2][card3] = i
            _flop_board_idx[card1][card3][card2] = i
            _flop_board_idx[card2][card1][card3] = i
            _flop_board_idx[card2][card3][card1] = i
            _flop_board_idx[card3][card1][card2] = i
            _flop_board_idx[card3][card2][card1] = i
    return out


# --- Gives the number of possible boards.
# -- @return the number of possible boards
def get_next_boards_count(street):
    used_cards = game_settings.board_card_count[street - 1]
    new_cards = game_settings.board_card_count[street] - game_settings.board_card_count[street - 1]
    return comb(game_settings.card_count - used_cards, new_cards)


# --- Gives all possible sets of board cards for the game.
# -- @return an NxK tensor, where N is the number of possible boards, and K is
# -- the number of cards on each board
def get_last_round_boards(board):
    board_index = 0

    street = board_to_street(board)
    boards_count = get_last_boards_count(street)
    out = arguments.Tensor(boards_count, game_settings.board_card_count[constants.streets_count - 1]).fill_(0.0)
    boards = arguments.Tensor(boards_count, game_settings.board_card_count[constants.streets_count - 1]).fill_(0.0)
    cur_board = arguments.Tensor(game_settings.board_card_count[constants.streets_count - 1]).fill_(0.0)

    if board.dim() > 0:
        for i in range(board.size(0)):
            cur_board[i] = board[i]

    _, out = _build_boards(boards, board_index, cur_board, out,
                           game_settings.board_card_count[street - 1] + 1,
                           game_settings.board_card_count[constants.streets_count - 1],
                           game_settings.board_card_count[street - 1] + 1)
    return out


# --- Gives the number of possible boards.
# -- @return the number of possible boards
def get_last_boards_count(street):
    used_cards = game_settings.board_card_count[street - 1]
    new_cards = game_settings.board_card_count[constants.streets_count - 1] - game_settings.board_card_count[street - 1]
    return comb(game_settings.card_count - used_cards, new_cards)


# --- Gives a numerical index for a set of board cards.
# -- @param board a non-empty vector of board cards
# -- @return the numerical index for the board
def get_board_index(board):
    assert board.size(0) > 3

    used_cards = arguments.Tensor(game_settings.card_count).fill_(0)
    for i in range(board.size(0)):
        used_cards[board[i]] = 1
    ans = 0
    for i in range(game_settings.card_count):
        if used_cards[i] == 0 :
            ans = ans + 1
        if i == board[-1]:
            return ans
    return -1


def get_file_range(file_name):
    out = arguments.Tensor(game_settings.hand_count).fill_(0)
    with open(file_name, 'r') as file_obj:
        lines = file_obj.readlines()
        for line in lines:
            line = line.replace('t', 'T')
            line = line.replace('j', 'J')
            line = line.replace('q', 'Q')
            line = line.replace('k', 'K')
            line = line.replace('a', 'A')
            parts = line.split()
            hand_string = parts[0]
            val = float(parts[1])
            if hand_string is not None and val is not None:
                hand = card_to_string.string_to_board(hand_string)
                if hand[0] > hand[1]:
                    temp = hand[0].item()
                    hand[0] = hand[1].item()
                    hand[1] = temp
                idx = get_hole_index([hand[0], hand[1]])
                out[idx] = val
    out.div_(out.sum())
    return out


# --- Gives a range vector that has uniform probability on each hand which is
# -- valid with a given board.
# -- @param board a possibly empty vector of board cards
# -- @return a range vector where invalid hands have 0 probability and valid
# -- hands have uniform probability
def get_uniform_range(board):
    out = get_possible_hand_indexes(board)
    out.div_(out.sum())
    return out


# --- Normalizes a range vector over hands which are valid with a given board.
# -- @param board a possibly empty vector of board cards
# -- @param range a range vector
# -- @return a modified version of `range` where each invalid hand is given 0
# -- probability and the vector is normalized
def normalize_range(board, range):
    mask = get_possible_hand_indexes(board)
    out = range.clone().mul_(mask)
    # return zero range if it all collides with board (avoid div by zero)
    if out.sum() == 0:
        return out
    out.div_(out.sum())
    return out


def _build_boards(boards, board_index, cur_board, out, card_index, last_index, base_index):
    if card_index == last_index + 1:
        for i in range(last_index):
            boards[board_index][i] = cur_board[i]

        out[board_index].copy_(cur_board)
        board_index += 1
        return board_index, out

    start_index = 0
    if card_index > base_index:
        start_index = int((cur_board[card_index - 2] + 1).item())

    for i in range(start_index, game_settings.card_count):
        good = True
        for j in range(card_index - 1):
            if cur_board[j] == i:
                good = False
        if good:
            cur_board[card_index - 1] = i
            board_index, out = _build_boards(boards, board_index, cur_board, out, card_index + 1, last_index, base_index)
    return board_index, out
