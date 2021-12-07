import gc
import math

import torch

import settings.arguments as arguments
import settings.constants as constants
import settings.game_settings as game_settings

import game.card_tools as card_tools
from game.evaluation.evaluator import Evaluator


class TerminalEquity(object):
    board: arguments.Tensor
    _pf_equity: arguments.Tensor = None
    _block_matrix: arguments.Tensor = None
    fold_matrix: arguments.Tensor

    def __init__(self):
        self.batch_size = 24        # increased to better go with num_board of 48 (turn) or 2352 (flop)
        self.matrix_mem = arguments.Tensor()

        arguments.logger.trace("Loading base equity data")
        arguments.timer.split_start("Loading block matrix", log_level="TRACE")
        self._block_matrix = torch.load("./terminal_equity/block_matrix.pt").type(arguments.Tensor)
        arguments.timer.split_stop("Block Matrix loaded in", log_level="LOADING")

        arguments.timer.split_start("Loading equity matrix", log_level="TRACE")
        self._pf_equity = torch.load("./terminal_equity/preflop_equity.pt").type(arguments.Tensor)
        arguments.timer.split_stop("Equity Matrix loaded in", log_level="LOADING")

    # --- Sets the board cards for the evaluator and creates its internal data structures.
    # -- @param board a possibly empty vector of board cards
    def set_board(self, board):
        self.board = board
        self._set_call_matrix(board)
        self._set_fold_matrix(board)

    # --- Computes (a batch of) counterfactual values that a player achieves at a terminal node
    # -- where no player has folded.
    # --
    # -- @{set_board} must be called before this function.
    # --
    # -- @param ranges a batch of opponent ranges in an NxK tensor, where N is the batch size
    # -- and K is the range size
    # -- @param result a NxK tensor in which to save the cfvs
    def call_value(self, ranges, result):
        torch.mm(ranges, self.equity_matrix, out=result)

    #
    # --- Computes (a batch of) counterfactual values that a player achieves at a terminal node
    # -- where a player has folded.
    # --
    # -- @{set_board} must be called before this function.
    # --
    # -- @param ranges a batch of opponent ranges in an NxK tensor, where N is the batch size
    # -- and K is the range size
    # -- @param result A NxK tensor in which to save the cfvs. Positive cfvs are returned, and
    # -- must be negated if the player in question folded.
    def fold_value(self, ranges, result):
        torch.mm(ranges, self.fold_matrix, out=result)

    # --- Constructs the matrix that turns player ranges into showdown equity.
    # --
    # -- Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay` is the equity
    # -- for the first player when no player folds.
    # --
    # -- @param board_cards a non-empty vector of board cards
    # -- @param call_matrix a tensor where the computed matrix is stored
    # -- @local
    def get_last_round_call_matrix(self, board_cards, call_matrix):
        assert board_cards.dim() == 0 or board_cards.size(0) == 1 or board_cards.size(0) == 2 or board_cards.size(
            0) == 5, 'Only Leduc, extended Leduc, and Texas Holdem are supported ' + board_cards.size(0)

        strength = Evaluator.batch_eval_fast(board_cards)
        strength_view_1 = strength.view(game_settings.hand_count, 1).expand_as(call_matrix)
        strength_view_2 = strength.view(1, game_settings.hand_count).expand_as(call_matrix)

        call_matrix.copy_(torch.gt(strength_view_1, strength_view_2))
        call_matrix.sub_(torch.lt(strength_view_1, strength_view_2).type_as(call_matrix))

        self._handle_blocking_cards(call_matrix, board_cards)

    # -- Constructs the matrix that turns player ranges into showdown equity.
    # --
    # -- Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay` is the equity
    # -- for the first player when no player folds.
    # --
    # -- @param board_cards a non-empty vector of board cards
    # -- @param call_matrix a tensor where the computed matrix is stored
    # -- @local
    def get_inner_call_matrix(self, board_cards, call_matrix):
        assert board_cards.dim() == 0 or board_cards.size(1) == 1 or board_cards.size(1) == 2 or board_cards.size(1) == 5, 'Only Leduc, extended Leduc, and Texas Holdem are supported'

        strength = Evaluator.batch_eval_fast(board_cards)
        num_boards = board_cards.size(0)

        # handling hand strengths (winning probs);
        strength_view_1 = strength.view(num_boards, game_settings.hand_count, 1).expand(num_boards, game_settings.hand_count, game_settings.hand_count)
        strength_view_2 = strength.view(num_boards, 1, game_settings.hand_count).expand_as(strength_view_1)
        possible_mask = torch.lt(strength, 0.0).type_as(call_matrix)

        for i in range(0, num_boards, self.batch_size):
            indices = slice(i, i + self.batch_size)
            sz = self.batch_size

            if i + self.batch_size > num_boards:
                indices = slice(i, num_boards)
                sz = num_boards - i

            self.matrix_mem = arguments.Tensor(sz, game_settings.hand_count, game_settings.hand_count)

            self.matrix_mem[0:sz].copy_(torch.gt(strength_view_1[indices], strength_view_2[indices]))
            self.matrix_mem[0:sz].mul_(possible_mask[indices].view(sz, 1, game_settings.hand_count).expand(sz, game_settings.hand_count, game_settings.hand_count))
            self.matrix_mem[0:sz].mul_(possible_mask[indices].view(sz, game_settings.hand_count, 1).expand(sz, game_settings.hand_count, game_settings.hand_count))
            call_matrix.add_(torch.sum(self.matrix_mem[0:sz], 0))

            self.matrix_mem[0:sz].copy_(torch.lt(strength_view_1[indices], strength_view_2[indices]))
            self.matrix_mem[0:sz].mul_(possible_mask[indices].view(sz, 1, game_settings.hand_count).expand(sz, game_settings.hand_count, game_settings.hand_count))
            self.matrix_mem[0:sz].mul_(possible_mask[indices].view(sz, game_settings.hand_count, 1).expand(sz, game_settings.hand_count, game_settings.hand_count))
            call_matrix.sub_(torch.sum(self.matrix_mem[0:sz], 0))

        self._handle_blocking_cards(call_matrix, board_cards)

        # proper clean-up of very large objects
        del strength
        del possible_mask
        del self.matrix_mem
        self.matrix_mem = arguments.Tensor()
        gc.collect()
        if arguments.use_gpu:
            torch.cuda.empty_cache()

    def get_hand_strengths(self):
        a = arguments.Tensor(1, game_settings.hand_count).fill_(1)
        return torch.mm(a, self.equity_matrix)

    # --- Zeroes entries in an equity matrix that correspond to invalid hands.
    # --
    # -- A hand is invalid if it shares any cards with the board.
    # --
    # -- @param equity_matrix the matrix to modify
    # -- @param board a possibly empty vector of board cards
    # -- @local
    def _handle_blocking_cards(self, equity_matrix, board: arguments.Tensor):
        possible_hand_indexes = card_tools.get_possible_hand_indexes(board)
        possible_hand_matrix = possible_hand_indexes.view(1, game_settings.hand_count).expand_as(equity_matrix)

        equity_matrix.mul_(possible_hand_matrix)
        possible_hand_matrix = possible_hand_indexes.view(game_settings.hand_count, 1).expand_as(equity_matrix)
        equity_matrix.mul_(possible_hand_matrix)

        if game_settings.hand_card_count == 2:
            equity_matrix.mul_(self._block_matrix)
        elif game_settings.hand_card_count == 1:
            for i in range(0, game_settings.card_count):
                equity_matrix[i][i] = 0

    # --- Sets the evaluator's call matrix, which gives the equity for terminal
    # -- nodes where no player has folded.
    # --
    # -- For nodes in the last betting round, creates the matrix `A` such that for player ranges
    # -- `x` and `y`, `x'Ay` is the equity for the first player when no player folds. For nodes
    # -- in the first betting round, gives the weighted average of all such possible matrices.
    # --
    # -- @param board a possibly empty vector of board cards
    # -- @local
    # -- TODO finish this
    def _set_call_matrix(self, board):
        street = card_tools.board_to_street(board)

        self.equity_matrix = arguments.Tensor(game_settings.hand_count, game_settings.hand_count).zero_()
        if street == constants.streets_count:
            # for last round we just return the matrix
            self.get_last_round_call_matrix(board, self.equity_matrix)
        elif street == 3 or street == 2:
            # iterate through all possible next round streets
            # TODO (go to the last street)
            next_round_boards = card_tools.get_last_round_boards(board)
            boards_count = next_round_boards.size(0)

            if self.matrix_mem.dim() != 3 or self.matrix_mem.size(1) != game_settings.hand_count or self.matrix_mem.size(2) != game_settings.hand_count:
                self.matrix_mem = arguments.Tensor(self.batch_size, game_settings.hand_count, game_settings.hand_count)

            self.get_inner_call_matrix(next_round_boards, self.equity_matrix)

            # averaging the values in the call matrix
            cards_to_come = game_settings.board_card_count[constants.streets_count - 1] - game_settings.board_card_count[street - 1]
            cards_left = game_settings.card_count - game_settings.hand_card_count * 2 - game_settings.board_card_count[street - 1]
            den = math.comb(cards_left, cards_to_come)
            weight_constant = 1/den
            self.equity_matrix.mul_(weight_constant)
        elif street == 1:
            self.equity_matrix.copy_(self._pf_equity)
        else:
            # impossible street
            assert False, 'impossible street ' + street

    # --- Sets the evaluator's fold matrix, which gives the equity for terminal
    # -- nodes where one player has folded.
    # --
    # -- Creates the matrix `B` such that for player ranges `x` and `y`, `x'By` is the equity
    # -- for the player who doesn't fold
    # -- @param board a possibly empty vector of board cards
    # -- @local
    def _set_fold_matrix(self, board):
        self.fold_matrix = arguments.Tensor(game_settings.hand_count, game_settings.hand_count)
        self.fold_matrix.fill_(1)
        # setting cards that block each other to zero
        self._handle_blocking_cards(self.fold_matrix, board)




