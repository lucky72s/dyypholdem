import math

import torch
import torch.nn

import settings.arguments as arguments
import settings.game_settings as game_settings
import settings.constants as constants

import nn.bucketer as bucketer
import game.card_tools as card_tools


class NextRoundValuePre(object):
    iter: int
    pot_sizes: torch.Tensor
    batch_size: int
    weight_constant: float

    next_round_inputs: torch.Tensor
    next_round_values: torch.Tensor
    transposed_next_round_values: torch.Tensor
    next_round_extended_range: torch.Tensor
    next_round_serialized_range: torch.Tensor
    range_normalization: torch.Tensor
    value_normalization: torch.Tensor
    range_normalization_memory: torch.Tensor
    counterfactual_value_memory: torch.Tensor

    bucket_range_on_board: torch.Tensor
    range_normalization_on_board: torch.Tensor
    value_normalization_on_board: torch.Tensor
    next_round_extended_range_on_board: torch.Tensor
    next_round_serialized_range_on_board: torch.Tensor
    next_round_inputs_on_board: torch.Tensor
    next_round_values_on_board: torch.Tensor
    values_per_board: torch.Tensor

    def __init__(self, nn, aux_nn, board):
        self.nn = nn
        self.aux_nn = aux_nn
        self._init_bucketing(board)

    # --- Initializes the tensor that translates hand ranges to bucket ranges.
    # -- @local
    def _init_bucketing(self, board):

        arguments.timer.start("Initializing pre-flop buckets", log_level="TRACE")

        street = card_tools.board_to_street(board)
        self._street = street
        self.bucket_count = bucketer.get_bucket_count(street + 1)
        boards = card_tools.get_next_round_boards(board)
        self.boards = boards
        self.board_count = boards.size(0)

        self.board_buckets = torch.load("./nn/bucketing/preflop_buckets.pt")
        if arguments.use_gpu:
            self.board_buckets = self.board_buckets.to('cuda')

        self.impossible_mask = torch.lt(self.board_buckets, 0)
        self.board_indexes = self.board_buckets.clone()
        self.board_indexes.masked_fill_(self.impossible_mask, 1)
        self.board_indexes_scatter = self.board_buckets.clone()
        self.board_indexes_scatter.masked_fill_(self.impossible_mask, self.bucket_count + 1)

        self.board_indexes = self.board_indexes.long()
        self.board_indexes_scatter = self.board_indexes_scatter.long()

        arguments.timer.stop(message="Pre-flop buckets initialized in", log_level="LOADING")

        # compute aux variables
        self.bucket_count_aux = bucketer.get_bucket_count(street)
        pf_buckets = bucketer.compute_buckets(arguments.Tensor([]))
        class_ids = arguments.Tensor()
        torch.arange(1, self.bucket_count_aux + 1, out=class_ids)

        class_ids = class_ids.view(1, self.bucket_count_aux).expand(game_settings.hand_count, self.bucket_count_aux)
        card_buckets = pf_buckets.view(game_settings.hand_count, 1).expand(game_settings.hand_count, self.bucket_count_aux)

        self._range_matrix_aux = arguments.Tensor(game_settings.hand_count, self.bucket_count_aux).zero_()
        self._range_matrix_aux[torch.eq(class_ids, card_buckets)] = 1
        self._reverse_value_matrix_aux = self._range_matrix_aux.t().clone()

        num_new_cards = game_settings.board_card_count[1] - game_settings.board_card_count[0]
        num_cur_cards = game_settings.board_card_count[0]
        den = math.comb(game_settings.card_count - num_cur_cards - 2 * game_settings.hand_card_count, num_new_cards)
        self.weight_constant = 1 / den

    # --- Initializes the value calculator with the pot size of each state that
    # -- we are going to evaluate.
    # --
    # -- During continual re-solving, there is one pot size for each initial state
    # -- of the second betting round (before board cards are dealt).
    # -- @param pot_sizes a vector of pot sizes
    # -- betting round ends
    def start_computation(self, pot_sizes, batch_size):
        self.iter = 0
        self.pot_sizes = pot_sizes.view(-1, 1).clone()
        self.pot_sizes = self.pot_sizes.expand(self.pot_sizes.size(0), batch_size).clone()
        self.pot_sizes = self.pot_sizes.view(-1, 1)
        self.batch_size = self.pot_sizes.size(0)

    def get_value_aux(self, ranges, values, next_board_idx):
        assert ranges.size(0) == self.batch_size, "ranges size does not match batch size"

        self.iter = self.iter + 1
        if self.iter == 1:
            # initializing data structures
            self.next_round_inputs = arguments.Tensor(self.batch_size, (self.bucket_count_aux * constants.players_count + 1)).zero_()
            self.next_round_values = arguments.Tensor(self.batch_size, constants.players_count, self.bucket_count_aux).zero_()
            self.next_round_extended_range = arguments.Tensor(self.batch_size, constants.players_count, self.bucket_count_aux).zero_()
            self.next_round_serialized_range = self.next_round_extended_range.view(-1, self.bucket_count_aux)
            self.range_normalization = arguments.Tensor()
            self.value_normalization = arguments.Tensor(self.batch_size, constants.players_count)

            # handling pot feature for the nn
            assert self._street <= 3
            den = game_settings.stack

            nn_bet_input = self.pot_sizes.clone().mul_(float(1 / den))
            self.next_round_inputs[:, -1:].copy_(nn_bet_input)

        # we need to find if we need remember something in this iteration
        use_memory = self.iter > arguments.cfr_skip_iters and next_board_idx is not None
        # logger.debug(f"Using memory={use_memory}")
        if use_memory and self.iter == arguments.cfr_skip_iters + 1:
            # first iter that we need to remember something - we need to init data structures
            self.bucket_range_on_board = arguments.Tensor(self.batch_size * constants.players_count, self.bucket_count)
            self.range_normalization_on_board = arguments.Tensor()
            self.value_normalization_on_board = arguments.Tensor(self.batch_size, constants.players_count)
            self.range_normalization_memory = arguments.Tensor(self.batch_size * constants.players_count, 1).zero_()
            self.counterfactual_value_memory = arguments.Tensor(self.batch_size, constants.players_count, self.bucket_count).zero_()
            self.next_round_extended_range_on_board = arguments.Tensor(self.batch_size, constants.players_count, self.bucket_count + 1).zero_()
            self.next_round_serialized_range_on_board = self.next_round_extended_range_on_board.view(-1, self.bucket_count + 1)
            self.next_round_inputs_on_board = arguments.Tensor(self.batch_size, (self.bucket_count * constants.players_count + 1)).zero_()
            self.next_round_values_on_board = arguments.Tensor(self.batch_size, constants.players_count, self.bucket_count).zero_()
            #  copy pot features over
            self.next_round_inputs_on_board[:, -1:].copy_(self.next_round_inputs[:, -1:])

        # computing bucket range in next street for both players at once
        self._card_range_to_bucket_range_aux(ranges.view(self.batch_size * constants.players_count, -1),
                                             self.next_round_extended_range.view(self.batch_size * constants.players_count, -1))

        self.range_normalization = torch.sum(self.next_round_serialized_range[:, 0:self.bucket_count_aux], 1)
        rn_view = self.range_normalization.view(self.batch_size, constants.players_count)
        for player in range(constants.players_count):
            self.value_normalization[:, player].copy_(rn_view[:, 1 - player])

        if use_memory:
            self._card_range_to_bucket_range_on_board(next_board_idx, ranges.view(self.batch_size * constants.players_count, -1),
                                                      self.next_round_extended_range_on_board.view(self.batch_size * constants.players_count, -1))
            self.range_normalization_on_board = torch.sum(self.next_round_serialized_range_on_board[:, 0:self.bucket_count], 1)
            rnb_view = self.range_normalization_on_board.view(self.batch_size, constants.players_count)
            for player in range(constants.players_count):
                self.value_normalization_on_board[:, player].copy_(rnb_view[:, 1 - player])
            self.range_normalization_memory.add_(self.value_normalization_on_board.view(self.range_normalization_memory.shape))

        # eliminating division by zero
        self.range_normalization[torch.eq(self.range_normalization, 0)] = 1
        self.next_round_serialized_range.div_(self.range_normalization.view(-1, 1).expand_as(self.next_round_serialized_range))
        for player in range(constants.players_count):
            player_range_index = [player * self.bucket_count_aux, (player + 1) * self.bucket_count_aux]
            self.next_round_inputs[:, player_range_index[0]:player_range_index[1]].copy_(self.next_round_extended_range[:, player, 0:self.bucket_count_aux])

        # using nn to compute values
        serialized_inputs_view = self.next_round_inputs.view(self.batch_size, -1)
        serialized_values_view = self.next_round_values.view(self.batch_size, -1)

        # computing value in the next round
        self.aux_nn.get_value(serialized_inputs_view, serialized_values_view)

        if use_memory:
            # eliminating division by zero
            self.range_normalization_on_board[torch.eq(self.range_normalization_on_board, 0)] = 1
            self.next_round_serialized_range_on_board.div_(self.range_normalization_on_board.view(-1, 1).expand_as(self.next_round_serialized_range_on_board))
            for player in range(constants.players_count):
                player_range_index = [player * self.bucket_count, (player + 1) * self.bucket_count]
                self.next_round_inputs_on_board[:, player_range_index[0]:player_range_index[1]].copy_(self.next_round_extended_range_on_board[:, player, 0:self.bucket_count])

            # using nn to compute values
            serialized_inputs_view_on_board = self.next_round_inputs_on_board.view(self.batch_size, -1)
            serialized_values_view_on_board = self.next_round_values_on_board.view(self.batch_size, -1)
            # computing value in the next round
            self.nn.get_value(serialized_inputs_view_on_board, serialized_values_view_on_board)

        # normalizing values back according to the orginal range sum
        normalization_view = self.value_normalization.view(self.batch_size, constants.players_count, 1)
        self.next_round_values.mul_(normalization_view.expand_as(self.next_round_values))

        if use_memory:
            normalization_view_on_board = self.value_normalization_on_board.view(self.batch_size, constants.players_count, 1)
            self.next_round_values_on_board.mul_(normalization_view_on_board.expand_as(self.next_round_values_on_board))
            self.counterfactual_value_memory.add_(self.next_round_values_on_board)

        # translating bucket values back to the card values
        self._bucket_value_to_card_value_aux(self.next_round_values.view(self.batch_size * constants.players_count, -1),
                                             values.view(self.batch_size * constants.players_count, -1))

    # --- Gives the predicted counterfactual values at each evaluated state, given
    # -- input ranges.
    # --
    # -- @{start_computation} must be called first. Each state to be evaluated must
    # -- be given in the same order that pot sizes were given for that function.
    # -- Keeps track of iterations internally, so should be called exactly once for
    # -- every iteration of continual re-solving.
    # --
    # -- @param ranges An Nx2xK tensor, where N is the number of states evaluated
    # -- (must match input to @{start_computation}), 2 is the number of players, and
    # -- K is the number of private hands. Contains N sets of 2 range vectors.
    # -- @param values an Nx2xK tensor in which to store the N sets of 2 value vectors
    # -- which are output
    def get_value(self, ranges, values):
        assert ranges.size(0) == self.batch_size, "ranges size does not match batch size"

        self.iter = self.iter + 1
        if self.iter == 1:
            # initializing data structures
            self.next_round_inputs = arguments.Tensor(self.batch_size, self.board_count, (self.bucket_count * constants.players_count + 1)).zero_()
            self.next_round_values = arguments.Tensor(self.batch_size, self.board_count, constants.players_count, self.bucket_count).zero_()
            self.transposed_next_round_values = arguments.Tensor(self.batch_size, constants.players_count, self.board_count, self.bucket_count)
            self.next_round_extended_range = arguments.Tensor(self.batch_size, constants.players_count, self.board_count, self.bucket_count + 1).zero_()
            self.next_round_serialized_range = self.next_round_extended_range.view(-1, self.bucket_count + 1)
            self.range_normalization = arguments.Tensor()
            self.value_normalization = arguments.Tensor(self.batch_size, constants.players_count, self.board_count)

            # handling pot feature for the nn
            assert self._street <= 3
            den = game_settings.stack

            # handling pot feature for the nn
            nn_bet_input = self.pot_sizes.clone().mul_(float(1 / den))
            nn_bet_input = nn_bet_input.view(-1, 1).expand(self.batch_size, self.board_count, 1)
            self.next_round_inputs[:, :, -1:].copy_(nn_bet_input)

        # computing bucket range in next street for both players at once
        self._card_range_to_bucket_range(ranges.view(self.batch_size * constants.players_count, -1),
                                         self.next_round_extended_range.view(self.batch_size * constants.players_count, -1))

        self.range_normalization = torch.sum(self.next_round_serialized_range[:, 0:self.bucket_count], 1)
        rn_view = self.range_normalization.view(self.batch_size, constants.players_count, self.board_count)
        for player in range(constants.players_count):
            self.value_normalization[:, player, :].copy_(rn_view[:, 1 - player, :])

        # eliminating division by zero
        self.range_normalization[torch.eq(self.range_normalization, 0)] = 1
        self.next_round_serialized_range.div_(self.range_normalization.view(-1, 1).expand_as(self.next_round_serialized_range))
        for player in range(constants.players_count):
            player_range_index = [player * self.bucket_count, (player + 1) * self.bucket_count]
            self.next_round_inputs[:, :, player_range_index[0]:player_range_index[1]].copy_(self.next_round_extended_range[:, player, :, 0:self.bucket_count])

        # using nn to compute values
        serialized_inputs_view = self.next_round_inputs.view(self.batch_size * self.board_count, -1)
        serialized_values_view = self.next_round_values.view(self.batch_size * self.board_count, -1)

        # computing value in the next round
        self.nn.get_value(serialized_inputs_view, serialized_values_view)

        # normalizing values back according to the orginal range sum
        normalization_view = self.value_normalization.view(self.batch_size, constants.players_count, self.board_count, 1).transpose(1, 2)
        self.next_round_values.mul_(normalization_view.expand_as(self.next_round_values))
        self.transposed_next_round_values.copy_(self.next_round_values.transpose(2, 1))

        # translating bucket values back to the card values
        self._bucket_value_to_card_value(self.transposed_next_round_values.view(self.batch_size * constants.players_count, -1),
                                         values.view(self.batch_size * constants.players_count, -1))

    # --- Gives the average counterfactual values on the given board across previous
    # -- calls to @{get_value}.
    # --
    # -- Used to update opponent counterfactual values during re-solving after board
    # -- cards are dealt.
    # -- @param board a non-empty vector of board cards
    # -- @param values a tensor in which to store the values
    def get_value_on_board(self, board, values):
        # check if we have evaluated correct number of iterations
        assert self.iter == arguments.cfr_iters

        batch_size = values.size(0)
        assert batch_size == self.batch_size

        self.range_normalization_memory[torch.eq(self.range_normalization_memory, 0)] = 1
        serialized_memory_view = self.counterfactual_value_memory.view(-1, self.bucket_count)
        serialized_memory_view.div_(self.range_normalization_memory.expand_as(serialized_memory_view))

        self._bucket_value_to_card_value_on_board(board, self.counterfactual_value_memory.view(self.batch_size * constants.players_count, -1),
                                                  values.view(self.batch_size * constants.players_count, -1))

    # -- Converts a range vector over private hands to a range vector over buckets.
    # -- @param card_range a probability vector over private hands
    # -- @param bucket_range a vector in which to store the output probabilities over buckets
    # @local
    def _card_range_to_bucket_range(self, card_range, bucket_range):
        other_bucket_range = bucket_range.view(-1, self.board_count, self.bucket_count + 1).zero_()
        indexes = self.board_indexes_scatter.view(1, self.board_count, game_settings.hand_count).expand(bucket_range.size(0), self.board_count, game_settings.hand_count)
        indexes = indexes.clone().sub_(1)  # subtract one as the buckets are 1-based indexed in the master file
        other_bucket_range.scatter_add_(2, indexes, card_range.view(-1, 1, game_settings.hand_count).expand(card_range.size(0), self.board_count, game_settings.hand_count))

    def _card_range_to_bucket_range_aux(self, card_range, bucket_range):
        torch.mm(card_range, self._range_matrix_aux, out=bucket_range)

    def _card_range_to_bucket_range_on_board(self, board_idx, card_range, bucket_range):
        other_bucket_range = bucket_range.view(-1, self.bucket_count + 1).zero_()
        indexes = self.board_indexes_scatter.view(1, self.board_count, game_settings.hand_count)[:, board_idx, :].expand(bucket_range.size(0), game_settings.hand_count)
        indexes = indexes.clone().sub_(1)  # subtract one as the buckets are 1-based indexed in the master file
        other_bucket_range.scatter_add_(1, indexes, card_range.view(-1, game_settings.hand_count).expand(card_range.size(0), game_settings.hand_count))

    # --- Converts a value vector over buckets to a value vector over private hands.
    # -- @param bucket_value a value vector over buckets
    # -- @param card_value a vector in which to store the output values over private hands
    # -- @local
    def _bucket_value_to_card_value(self, bucket_value, card_value):
        indexes = self.board_indexes.view(1, self.board_count, game_settings.hand_count).expand(bucket_value.size(0), self.board_count, game_settings.hand_count)
        indexes = indexes.clone().sub_(1)  # subtract one as the buckets are 1-based indexed in the master file
        self.values_per_board = bucket_value.view(bucket_value.size(0), self.board_count, self.bucket_count).gather(2, indexes)
        impossible = self.impossible_mask.view(1, self.board_count, game_settings.hand_count).expand(bucket_value.size(0), self.board_count, game_settings.hand_count)
        self.values_per_board.masked_fill_(impossible, 0)
        torch.sum(self.values_per_board, 1, out=card_value)
        card_value.mul_(self.weight_constant)

    def _bucket_value_to_card_value_aux(self, bucket_value, card_value):
        torch.mm(bucket_value, self._reverse_value_matrix_aux, out=card_value)

    # --- Converts a value vector over buckets to a value vector over private hands
    # -- given a particular set of board cards.
    # --
    # -- @param board a non-empty vector of board cards
    # -- @param bucket_value a value vector over buckets
    # -- @param card_value a vector in which to store the output values over private hands
    # -- @local
    def _bucket_value_to_card_value_on_board(self, board, bucket_value, card_value):
        board_idx = card_tools.get_flop_board_index(board)
        indexes = self.board_indexes.view(1, self.board_count, game_settings.hand_count)[:, board_idx, :].expand(bucket_value.size(0), game_settings.hand_count)
        indexes = indexes.clone().sub_(1)  # subtract one as the buckets are 1-based indexed in the master file
        self.values_per_board = bucket_value.view(bucket_value.size(0), self.bucket_count).gather(1, indexes)
        impossible = self.impossible_mask.view(1, self.board_count, game_settings.hand_count)[:, board_idx, :].expand(bucket_value.size(0), game_settings.hand_count)
        self.values_per_board.masked_fill_(impossible, 0)
        card_value.copy_(self.values_per_board)
