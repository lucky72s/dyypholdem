import math

import torch

import settings.arguments as arguments
import settings.game_settings as game_settings

import game.card_tools as card_tools


class Evaluator(object):

    _texas_lookup: torch.Tensor
    _idx_to_cards = torch.zeros(game_settings.hand_count, game_settings.hand_card_count)

    for card1 in range(0, game_settings.card_count):
        for card2 in range(card1 + 1, game_settings.card_count):
            idx = card_tools.get_hole_index([card1, card2])
            _idx_to_cards[idx][0] = card1
            _idx_to_cards[idx][1] = card2

    arguments.timer.start("Loading hand ranks lookup table...", log_level="TRACE")

    _texas_lookup = torch.load("./game/evaluation/hand_ranks.pt").type(arguments.LongTensor)

    arguments.timer.stop(message="Hand ranks lookup table initialized in:", log_level="TIMING")

    # --- Gives a strength representation for a hand containing two cards.
    # -- @param hand_ranks the rank of each card in the hand
    # -- @return the strength value of the hand
    # -- @local
    @classmethod
    def evaluate_two_card_hand(cls, hand_ranks):
        raise NotImplementedError()

    # --- Gives a strength representation for a hand containing three cards.
    # -- @param hand_ranks the rank of each card in the hand
    # -- @return the strength value of the hand
    # -- @local
    @classmethod
    def evaluate_three_card_hand(cls, hand_ranks):
        raise NotImplementedError()

    # --- Gives a strength representation for a texas hold'em hand containing seven cards.
    # -- @param hand_ranks the rank of each card in the hand
    # -- @return the strength value of the hand
    # -- @local
    @classmethod
    def evaluate_seven_card_hand(cls, hand):
        rank = cls._texas_lookup[54 + (hand[0] - 1) + 1]
        for c in range(1, hand.size(0)):
            rank = cls._texas_lookup[1 + rank + (hand[c] - 1) + 1]
        return -rank

    # --- Gives a strength representation for a two or three card hand.
    # -- @param hand a vector of two or three cards
    # -- @param[opt] impossible_hand_value the value to return if the hand is invalid
    # -- @return the strength value of the hand, or `impossible_hand_value` if the
    # -- hand is invalid
    @classmethod
    def evaluate(cls, hand: torch.Tensor, impossible_hand_value):
        assert hand.max() < game_settings.card_count and hand.min() >= 0, 'hand does not correspond to any cards'

        impossible_hand_value = impossible_hand_value or -1
        if not card_tools.hand_is_possible(hand):
            return impossible_hand_value

        # we are not interested in the hand suit - we will use ranks instead of cards
        if hand.size(0) == 2:
            raise NotImplementedError()
        elif hand.size(0) == 3:
            raise NotImplementedError()
        elif hand.size(0) == 7:
            return cls.evaluate_seven_card_hand(hand)
        else:
            assert False, 'unsupported size of hand!'

    @classmethod
    def evaluate_fast(cls, hands):
        ret = cls._texas_lookup.index_select(0, torch.add(hands[:, 0], 54))
        for c in range(1, hands.size(1)):
            ret = cls._texas_lookup.index_select(0, torch.add(hands[:, c], ret.long()).add(1))
        ret.mul_(card_tools.get_possible_hands_mask(hands))
        ret.mul_(-1)
        return ret

    # --- Gives strength representations for all private hands on the given board.
    # -- @param board a possibly empty vector of board cards
    # -- @param impossible_hand_value the value to assign to hands which are invalid
    # -- on the board
    # -- @return a vector containing a strength value or `impossible_hand_value` for
    # -- every private hand
    @classmethod
    def batch_eval(cls, board, impossible_hand_value):
        hand_values = arguments.Tensor(game_settings.hand_count).fill_(-1)
        if board.dim() == 0: # Kuhn poker
            for hand in range(0, game_settings.card_count):
                hand_values[hand] = math.floor(hand / game_settings.suit_count) + 1
        else:
            board_size = board.size(0)
            assert board_size == 1 or board_size == 2 or board_size == 5, 'Incorrect board size'

            whole_hand = arguments.Tensor(board_size + game_settings.hand_card_count)
            whole_hand[0: -game_settings.hand_card_count].copy_(board)
            whole_hand = whole_hand.int()

            if game_settings.hand_card_count == 1:
                for card in range(0, game_settings.card_count):
                    whole_hand[-1] = card
                    hand_values[card] = cls.evaluate(whole_hand, impossible_hand_value)
            elif game_settings.hand_card_count == 2:
                for card1 in range(0, game_settings.card_count):
                    for card2 in range(card1 + 1, game_settings.card_count):
                        whole_hand[-2] = card1
                        whole_hand[-1] = card2
                        idx = card_tools.get_hole_index([card1, card2])
                        hand_values[idx] = cls.evaluate(whole_hand, impossible_hand_value)
            else:
                assert False, "unsupported hand_card_count: " + game_settings.hand_card_count
        return hand_values

    @classmethod
    def batch_eval_fast(cls, board: arguments.Tensor):
        if board.dim() == 0:  # -- kuhn poker
            return None
        elif board.dim() == 2:
            batch_size = board.size(0)
            hands = arguments.Tensor(batch_size, game_settings.hand_count, board.size(1) + game_settings.hand_card_count).long()
            hands[:, :, 0: board.size(1)].copy_(board.view(batch_size, 1, board.size(1)).expand(batch_size, game_settings.hand_count, board.size(1)))
            hands[:, :, -2:].copy_(cls._idx_to_cards.view(1, game_settings.hand_count, game_settings.hand_card_count).expand(batch_size, game_settings.hand_count, game_settings.hand_card_count))
            return cls.evaluate_fast(hands.view(-1, board.size(1) + game_settings.hand_card_count)).view(batch_size, game_settings.hand_count)
        elif board.dim() == 1:
            hands = arguments.Tensor(game_settings.hand_count, board.size(0) + game_settings.hand_card_count).long()
            hands[:, 0: board.size(0)].copy_(board.view(1, board.size(0)).expand(game_settings.hand_count, board.size(0)))
            hands[:, -2:].copy_(cls._idx_to_cards)
            return cls.evaluate_fast(hands)
        else:
            assert False, "weird board dim " + board.dim()
