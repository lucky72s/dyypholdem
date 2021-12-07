import math

import torch

import settings.arguments as arguments
import settings.game_settings as game_settings

import game.card_tools as card_tools
import nn.bucketer as bucketer


class BucketConversion(object):

    _bucket_count: int
    _range_matrix: torch.Tensor
    _reverse_value_matrix: torch.Tensor

    def __init__(self):
        pass

    # --- Sets the board cards for the bucketer.
    # -- @param board a non-empty vector of board cards
    def set_board(self, board, raw=None):

        if raw is not None:
            self._bucket_count = math.comb(14, 2) + math.comb(10, 2)
        else:
            self._bucket_count = bucketer.get_bucket_count(card_tools.board_to_street(board))

        self._range_matrix = arguments.Tensor(game_settings.hand_count, self._bucket_count).zero_()

        buckets = None
        if raw is not None:
            buckets = bucketer.compute_rank_buckets(board)
        else:
            buckets = bucketer.compute_buckets(board)

        class_ids = arguments.Tensor()
        torch.arange(1, self._bucket_count + 1, out=class_ids)

        class_ids = class_ids.view(1, self._bucket_count).expand(game_settings.hand_count, self._bucket_count)
        card_buckets = buckets.view(game_settings.hand_count, 1).expand(game_settings.hand_count, self._bucket_count)

        # finding all strength classes
        # matrix for transformation from card ranges to strength class ranges
        self._range_matrix[torch.eq(class_ids, card_buckets)] = 1

        # matrix for transformation form class values to card values
        self._reverse_value_matrix = self._range_matrix.t().clone()

    # --- Converts a range vector over private hands to a range vector over buckets.
    # --
    # -- @{set_board} must be called first. Used to create inputs to the neural net.
    # -- @param card_range a probability vector over private hands
    # -- @param bucket_range a vector in which to save the resulting probability
    # -- vector over buckets
    def card_range_to_bucket_range(self, card_range, bucket_range):
        torch.mm(card_range, self._range_matrix, out=bucket_range)

    def hand_cfvs_to_bucket_cfvs(self, card_range, card_cfvs, bucket_range, bucketed_cfvs):
        torch.mm(torch.mul(card_range, card_cfvs), self._range_matrix, out=bucketed_cfvs)
        # avoid divide by 0
        bucketed_cfvs.div_(torch.clamp(bucket_range, min=0.00001))
