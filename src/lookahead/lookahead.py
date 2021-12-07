from typing import Dict, Union

import torch

import settings.arguments as arguments
import settings.game_settings as game_settings
import settings.constants as constants

from terminal_equity.terminal_equity import TerminalEquity
from lookahead.lookahead_builder import LookaheadBuilder
from lookahead.resolve_results import ResolveResults
from lookahead.cfrd_gadget import CFRDGadget
from tree.tree_node import TreeNode
from nn.next_round_value import NextRoundValue
from nn.next_round_value_pre import NextRoundValuePre


class Lookahead(object):
    reconstruction_gadget: CFRDGadget

    batch_size: int
    terminal_equity: TerminalEquity
    builder: LookaheadBuilder
    tree: TreeNode
    depth: int

    regret_epsilon: float
    acting_player: Dict[int, int]
    bets_count: dict
    nonallinbets_count: Dict[int, int]
    terminal_actions_count: dict
    actions_count: dict
    first_call_terminal: bool
    first_call_transition: bool
    first_call_check: bool

    pot_size: dict
    ranges_data: Dict[int, arguments.Tensor]
    average_strategies_data: dict
    current_strategy_data: dict
    cfvs_data: dict
    average_cfvs_data: dict
    regrets_data: Dict[int, arguments.Tensor]
    current_regrets_data: dict
    positive_regrets_data: Dict[int, arguments.Tensor]
    placeholder_data: Dict[int, arguments.Tensor]
    regrets_sum: Dict[int, arguments.Tensor]
    empty_action_mask: dict  # --used to mask empty actions

    action_to_index: dict
    next_street_boxes: Union[NextRoundValue, NextRoundValuePre]
    indices = dict
    num_pot_sizes: int
    next_street_boxes_inputs: torch.Tensor
    next_street_boxes_outputs: torch.Tensor
    next_board_idx: int
    next_round_pot_sizes: torch.Tensor

    term_call_indices: dict
    num_term_call_nodes: int
    term_fold_indices: dict
    num_term_fold_nodes: int
    ranges_data_call: arguments.Tensor
    ranges_data_fold: arguments.Tensor
    cfvs_data_call: arguments.Tensor
    cfvs_data_fold: arguments.Tensor

    # used to hold and swap inner (non-terminal) nodes when doing some transpose operations
    inner_nodes: Dict[int, arguments.Tensor]
    inner_nodes_p1: dict
    swap_data: dict

    def __init__(self, terminal_equity, batch_size):
        self.terminal_equity = terminal_equity
        self.batch_size = batch_size
        self.builder = LookaheadBuilder(self)

        self.reconstruction_opponent_cfvs = None
        self.next_board_idx = None

    # --- Constructs the lookahead from a game's public tree.
    # --
    # -- Must be called to initialize the lookahead.
    # -- @param tree a public tree
    def build_lookahead(self, tree):
        self.builder.build_from_tree(tree)

    def reset(self):
        self.builder.reset()

    # --- Re-solves the lookahead using input ranges.
    # --
    # -- Uses the input range for the opponent instead of a gadget range, so only
    # -- appropriate for re-solving the root node of the game tree (where ranges
    # -- are fixed).
    # --
    # -- @{build_lookahead} must be called first.
    # --
    # -- @param player_range a range vector for the re-solving player
    # -- @param opponent_range a range vector for the opponent
    def resolve_first_node(self, player_range, opponent_range):
        self.ranges_data[1][:, :, :, :, 0, :].copy_(player_range)
        self.ranges_data[1][:, :, :, :, 1, :].copy_(opponent_range)
        self._compute()

    # --- Re-solves the lookahead using an input range for the player and
    # -- the @{cfrd_gadget|CFRDGadget} to generate ranges for the opponent.
    # --
    # -- @{build_lookahead} must be called first.
    # --
    # -- @param player_range a range vector for the re-solving player
    # -- @param opponent_cfvs a vector of cfvs achieved by the opponent
    # -- before re-solving
    def resolve(self, player_range, opponent_cfvs):
        assert player_range is not None
        assert opponent_cfvs is not None

        self.reconstruction_gadget = CFRDGadget(self.tree.board, player_range, opponent_cfvs)
        self.ranges_data[1][:, :, :, :, 0, :].copy_(player_range)
        self.reconstruction_opponent_cfvs = opponent_cfvs
        self._compute()

    # --- Gets the results of re-solving the lookahead.
    # --
    # -- The lookahead must first be re-solved with @{resolve} or
    # -- @{resolve_first_node}.
    # --
    # -- @return a table containing the fields:
    # --
    # -- * `strategy`: an AxK tensor containing the re-solve player's strategy at the
    # -- root of the lookahead, where A is the number of actions and K is the range size
    # --
    # -- * `achieved_cfvs`: a vector of the opponent's average counterfactual values at the
    # -- root of the lookahead
    # --
    # -- * `children_cfvs`: an AxK tensor of opponent average counterfactual values after
    # -- each action that the re-solve player can take at the root of the lookahead
    def get_results(self) -> ResolveResults:

        out = ResolveResults()

        # 0.0 extract actions from tree
        out.actions = self.tree.actions.tolist()

        # 1.0 average strategy [actions x range]
        # lookahead already computes the average strategy we just convert the dimensions
        out.strategy = self.average_strategies_data[2].view(-1, self.batch_size, game_settings.hand_count).clone()

        # 2.0 achieved opponent's CFVs at the starting node
        out.achieved_cfvs = self.average_cfvs_data[1].view(self.batch_size, constants.players_count,
                                                           game_settings.hand_count)[:, 0, :].clone()

        # 3.0 CFVs for the acting player only when resolving first node
        if self.reconstruction_opponent_cfvs is not None:
            out.root_cfvs = None
        else:
            out.root_cfvs = self.average_cfvs_data[1].view(self.batch_size, constants.players_count,
                                                           game_settings.hand_count)[:, 1, :].clone()
            # swap cfvs indexing
            out.root_cfvs_both_players = self.average_cfvs_data[1].view(self.batch_size, constants.players_count,
                                                                        game_settings.hand_count).clone()
            out.root_cfvs_both_players[:, 1, :].copy_(
                self.average_cfvs_data[1].view(self.batch_size, constants.players_count, game_settings.hand_count)[:, 0,
                :])
            out.root_cfvs_both_players[:, 0, :].copy_(
                self.average_cfvs_data[1].view(self.batch_size, constants.players_count, game_settings.hand_count)[:, 1,
                :])

        # 4.0 children CFVs [actions x range]
        out.children_cfvs = self.average_cfvs_data[2][:, :, :, :, 0, :].clone().view(-1, game_settings.hand_count)

        # IMPORTANT divide average CFVs by average strategy in here
        scaler = self.average_strategies_data[2].view(-1, self.batch_size, game_settings.hand_count).clone()
        range_mul = self.ranges_data[1][:, :, :, :, 0, :].clone().view(1, self.batch_size,
                                                                       game_settings.hand_count).clone()
        range_mul = range_mul.expand_as(scaler)
        scaler = scaler.mul_(range_mul)
        scaler = scaler.sum(2, keepdim=True).expand_as(range_mul).clone()
        scaler = scaler.mul_(arguments.cfr_iters - arguments.cfr_skip_iters)
        out.children_cfvs.div_(scaler.view(out.children_cfvs.shape))

        assert out.children_cfvs is not None
        assert out.strategy is not None
        assert out.achieved_cfvs is not None

        return out

    # --- Gives the average counterfactual values for the opponent during re-solving
    # -- after a chance event (the betting round changes and more cards are dealt).
    # --
    # -- Used during continual re-solving to track opponent cfvs. The lookahead must
    # -- first be re-solved with @{resolve} or @{resolve_first_node}.
    # --
    # -- @param action_index the action taken by the re-solving player at the start
    # -- of the lookahead
    # -- @param board a tensor of board cards, updated by the chance event
    # -- @return a vector of cfvs
    def get_chance_action_cfv(self, action, board):
        box_outputs = self.next_street_boxes_outputs.view(-1, constants.players_count, game_settings.hand_count)
        next_street_box = self.next_street_boxes
        batch_index = self.action_to_index[action]
        assert batch_index is not None
        pot_mult = self.next_round_pot_sizes[batch_index]
        if box_outputs is None:
            assert False
        next_street_box.get_value_on_board(board, box_outputs)
        out = box_outputs[batch_index][self.tree.current_player.value]
        out.mul_(pot_mult)

        return out

    # --- Re-solves the lookahead.
    # -- @local
    def _compute(self):
        # 1.0 main loop
        for iteration in range(1, arguments.cfr_iters + 1):
            self._set_opponent_starting_range()
            self._compute_current_strategies()
            self._compute_ranges()
            self._compute_update_average_strategies(iteration)
            self._compute_terminal_equities()
            self._compute_cfvs()
            self._compute_regrets()
            self._compute_cumulate_average_cfvs(iteration)

        # 2.0 at the end normalize average strategy
        self._compute_normalize_average_strategies()
        # 2.1 normalize root's CFVs
        self._compute_normalize_average_cfvs()

    # --- Generates the opponent's range for the current re-solve iteration using
    # -- the @{cfrd_gadget|CFRDGadget}.
    # -- @param iteration the current iteration number of re-solving
    # -- @local
    def _set_opponent_starting_range(self):
        if self.reconstruction_opponent_cfvs is not None:
            # note that CFVs indexing is swapped, thus the CFVs for the reconstruction player are for player '1'
            opponent_range = self.reconstruction_gadget.compute_opponent_range(self.cfvs_data[1][:, :, :, :, 0, :])
            self.ranges_data[1][:, :, :, :, 1, :].copy_(opponent_range)

    # --- Uses regret matching to generate the players' current strategies.
    # -- @local
    def _compute_current_strategies(self):
        for d in range(2, self.depth + 1):
            self.positive_regrets_data[d].copy_(self.regrets_data[d])
            self.positive_regrets_data[d].clamp_(self.regret_epsilon, constants.max_number())

            # 1.0 set regret of empty actions to 0
            self.positive_regrets_data[d].mul_(self.empty_action_mask[d])

            # 1.1  regret matching
            # note that the regrets as well as the CFVs have switched player indexing
            self.regrets_sum[d] = torch.sum(self.positive_regrets_data[d], 0)
            self.current_strategy_data[d] = torch.div(self.positive_regrets_data[d], self.regrets_sum[d].expand_as(self.positive_regrets_data[d]))

    # --- Using the players' current strategies, computes their probabilities of
    # -- reaching each state of the lookahead.
    # -- @local
    def _compute_ranges(self):
        for d in range(1, self.depth):
            current_level_ranges = self.ranges_data[d]
            next_level_ranges = self.ranges_data[d + 1]
            prev_layer_terminal_actions_count = self.terminal_actions_count[d - 1]
            prev_layer_actions_count = self.actions_count[d - 1]
            prev_layer_bets_count = self.bets_count[d - 1]
            gp_layer_nonallin_bets_count = self.nonallinbets_count[d - 2]
            gp_layer_terminal_actions_count = self.terminal_actions_count[d - 2]

            # copy the ranges of inner nodes and transpose
            inner_nodes: torch.Tensor = current_level_ranges[prev_layer_terminal_actions_count:, :gp_layer_nonallin_bets_count, :, :, :, :]
            inner_nodes = inner_nodes.transpose(1, 2)
            inner_nodes = torch.reshape(inner_nodes, self.inner_nodes[d].shape)
            self.inner_nodes[d].copy_(inner_nodes)

            super_view = self.inner_nodes[d]
            super_view = super_view.view(1, prev_layer_bets_count, -1, self.batch_size, constants.players_count,
                                         game_settings.hand_count)
            super_view = super_view.expand_as(next_level_ranges)
            next_level_strategies = self.current_strategy_data[d + 1]

            next_level_ranges.copy_(super_view)

            # multiply the ranges of the acting player by his strategy
            next_level_ranges[:, :, :, :, self.acting_player[d] - 1, :].mul_(next_level_strategies)

    # --- Updates the players' average strategies with their current strategies.
    # -- @param iter the current iteration number of re-solving
    # -- @local
    def _compute_update_average_strategies(self, iteration):
        if iteration > arguments.cfr_skip_iters:
            # no need to go through layers since we care for the average strategy only in the first node anyway
            # note that if you wanted to average strategy on lower layers, you would need to weight the current strategy by the current reach probability
            self.average_strategies_data[2].add_(self.current_strategy_data[2])

    # --- Using the players' reach probabilities, computes their counterfactual
    # -- values at all terminal states of the lookahead.
    # --
    # -- These include terminal states of the game and depth-limited states.
    # -- @local
    def _compute_terminal_equities(self):
        if self.tree.street != constants.streets_count:
            self._compute_terminal_equities_next_street_box()

        self._compute_terminal_equities_terminal_equity()
        # multiply by pot scale factor
        for d in range(2, self.depth + 1):
            self.cfvs_data[d].mul_(self.pot_size[d])

    # --- Using the players' reach probabilities, calls the neural net to compute the
    # -- players' counterfactual values at the depth-limited states of the lookahead.
    # -- @local
    def _compute_terminal_equities_next_street_box(self):
        assert self.tree.street != constants.streets_count

        if self.num_pot_sizes == 0:
            return

        for d in range(2, self.depth + 1):
            if d > 2 or self.first_call_transition:
                # if there's only 1 parent, then it should've been an all in, so skip this next_street_box calculation
                if self.ranges_data[d][1].size(0) > 1 or (d == 2 and self.first_call_transition):
                    parent_indices = [0, self.ranges_data[d][1].size(0) - 1]
                    if d == 2:
                        parent_indices = [0, 1]
                    destination = self.next_street_boxes_outputs[self.indices[d][0]:self.indices[d][1], :, :, :]
                    source = self.ranges_data[d][1, parent_indices[0]:parent_indices[1], :, :, :, :].view(destination.shape)
                    destination.copy_(source)

        if self.tree.current_player == constants.Players.P2:
            self.next_street_boxes_inputs.copy_(self.next_street_boxes_outputs)
        else:
            self.next_street_boxes_inputs[:, :, 0, :].copy_(self.next_street_boxes_outputs[:, :, 1, :])
            self.next_street_boxes_inputs[:, :, 1, :].copy_(self.next_street_boxes_outputs[:, :, 0, :])

        if self.tree.street == 1:
            self.next_street_boxes.get_value_aux(self.next_street_boxes_inputs.view(-1, constants.players_count, game_settings.hand_count),
                                                 self.next_street_boxes_outputs.view(-1, constants.players_count, game_settings.hand_count), self.next_board_idx)
        else:
            self.next_street_boxes.get_value(self.next_street_boxes_inputs.view(-1, constants.players_count, game_settings.hand_count),
                                             self.next_street_boxes_outputs.view(-1, constants.players_count, game_settings.hand_count))

        # now the neural net outputs for P1 and P2 respectively, so we need to swap the output values if necessary
        if self.tree.current_player == constants.Players.P2:
            self.next_street_boxes_inputs.copy_(self.next_street_boxes_outputs)
            self.next_street_boxes_outputs[:, :, 0, :].copy_(self.next_street_boxes_inputs[:, :, 1, :])
            self.next_street_boxes_outputs[:, :, 1, :].copy_(self.next_street_boxes_inputs[:, :, 0, :])

        for d in range(2, self.depth + 1):
            if d > 2 or self.first_call_transition:
                if self.ranges_data[d][1].size(0) > 1 or (d == 2 and self.first_call_transition):
                    parent_indices = [0, self.cfvs_data[d][1].size(0) - 1]
                    if d == 2:
                        parent_indices = [0, 1]
                    destination = self.cfvs_data[d][1, parent_indices[0]:parent_indices[1], :, :, :, :]
                    source = self.next_street_boxes_outputs[self.indices[d][0]:self.indices[d][1], :, :, :].view(destination.shape)
                    destination.copy_(source)

    # --- Using the players' reach probabilities, computes their counterfactual
    # -- values at each lookahead state which is a terminal state of the game.
    # -- @local
    def _compute_terminal_equities_terminal_equity(self):
        # copy in range data
        call_indices = []
        fold_indices = []
        for d in range(2, self.depth + 1):
            if d > 2 or self.first_call_terminal:
                if self.tree.street != constants.streets_count:
                    call_indices = self.term_call_indices[d]
                    target = self.ranges_data_call[call_indices[0]:call_indices[1]]
                    target.copy_(self.ranges_data[d][1][-1].view(target.shape))
                else:
                    call_indices = self.term_call_indices[d]
                    target = self.ranges_data_call[call_indices[0]:call_indices[1]]
                    target.copy_(self.ranges_data[d][1].view(target.shape))
            fold_indices = self.term_fold_indices[d]
            target = self.ranges_data_fold[fold_indices[0]:fold_indices[1]]
            target.copy_(self.ranges_data[d][0].view(target.shape))

        self.terminal_equity.call_value(self.ranges_data_call.view(-1, game_settings.hand_count),
                                        self.cfvs_data_call.view(-1, game_settings.hand_count))

        self.terminal_equity.fold_value(self.ranges_data_fold.view(-1, game_settings.hand_count),
                                        self.cfvs_data_fold.view(-1, game_settings.hand_count))

        for d in range(2, self.depth + 1):
            if self.tree.street != constants.streets_count:
                if d > 2 or self.first_call_terminal:
                    call_indices = self.term_call_indices[d]
                    self.cfvs_data[d][1][-1].copy_(
                        self.cfvs_data_call[call_indices[0]:call_indices[1]].view(self.cfvs_data[d][1][-1].shape))
            else:
                if d > 2 or self.first_call_terminal:
                    call_indices = self.term_call_indices[d]
                    self.cfvs_data[d][1].copy_(
                        self.cfvs_data_call[call_indices[0]:call_indices[1]].view(self.cfvs_data[d][1].shape))

            fold_indices = self.term_fold_indices[d]
            self.cfvs_data[d][0].copy_(
                self.cfvs_data_fold[fold_indices[0]:fold_indices[1]].view(self.cfvs_data[d][0].shape))

            # correctly set the folded player by multiplying by -1
            fold_multiplier = (self.acting_player[d] * 2 - 3)
            self.cfvs_data[d][0, :, :, :, 0, :].mul_(fold_multiplier)
            self.cfvs_data[d][0, :, :, :, 1, :].mul_(-fold_multiplier)

    # --- Using the players' reach probabilities and terminal counterfactual
    # -- values, computes their cfvs at all states of the lookahead.
    # -- @local
    def _compute_cfvs(self):
        for d in range(self.depth, 1, -1):
            gp_layer_terminal_actions_count = self.terminal_actions_count[d - 2]
            ggp_layer_nonallin_bets_count = self.nonallinbets_count[d - 3]

            self.cfvs_data[d][:, :, :, :, 0, :].mul_(self.empty_action_mask[d])
            self.cfvs_data[d][:, :, :, :, 1, :].mul_(self.empty_action_mask[d])

            self.placeholder_data[d].copy_(self.cfvs_data[d])

            # player indexing is swapped for cfvs
            self.placeholder_data[d][:, :, :, :, self.acting_player[d] - 1, :].mul_(self.current_strategy_data[d])

            self.regrets_sum[d] = torch.sum(self.placeholder_data[d], 0)

            # use a swap placeholder to change {{1,2,3}, {4,5,6}} into {{1,2}, {3,4}, {5,6}}
            swap = self.swap_data[d - 1]
            swap.copy_(self.regrets_sum[d].view(swap.shape))

            self.cfvs_data[d - 1][gp_layer_terminal_actions_count:, 0:ggp_layer_nonallin_bets_count, :, :, :, :].copy_(
                swap.transpose(1, 2))

    # --- Using the players' counterfactual values, updates their total regrets
    # -- for every state in the lookahead.
    # -- @local
    def _compute_regrets(self):
        for d in range(self.depth, 1, -1):
            gp_layer_terminal_actions_count = self.terminal_actions_count[d - 2]
            gp_layer_bets_count = self.bets_count[d - 2]
            ggp_layer_nonallin_bets_count = self.nonallinbets_count[d - 3]

            current_regrets = self.current_regrets_data[d]
            current_regrets.copy_(self.cfvs_data[d][:, :, :, :, self.acting_player[d] - 1, :])

            next_level_cfvs = self.cfvs_data[d - 1]

            parent_inner_nodes = self.inner_nodes_p1[d - 1]

            cfvs = next_level_cfvs[gp_layer_terminal_actions_count:, 0:ggp_layer_nonallin_bets_count, :, :, self.acting_player[d] - 1, :]
            cfvs = cfvs.transpose(1, 2)
            cfvs = torch.reshape(cfvs, parent_inner_nodes.shape)
            parent_inner_nodes.copy_(cfvs)
            parent_inner_nodes = parent_inner_nodes.view(1, gp_layer_bets_count, -1, self.batch_size,
                                                         game_settings.hand_count)
            parent_inner_nodes = parent_inner_nodes.expand_as(current_regrets)
            current_regrets.sub_(parent_inner_nodes)
            self.regrets_data[d].add_(current_regrets)

            # (CFR+)
            self.regrets_data[d].clamp_(0, constants.max_number())

    # --- Updates the players' average counterfactual values with their cfvs from the
    # -- current iteration.
    # -- @param iter the current iteration number of re-solving
    # -- @local
    def _compute_cumulate_average_cfvs(self, iteration):
        if iteration > arguments.cfr_skip_iters:
            self.average_cfvs_data[1].add_(self.cfvs_data[1])
            self.average_cfvs_data[2].add_(self.cfvs_data[2])

    # --- Normalizes the players' average strategies.
    # --
    # -- Used at the end of re-solving so that we can track un-normalized average
    # -- strategies, which are simpler to compute.
    # -- @local
    def _compute_normalize_average_strategies(self):
        # using regrets_sum as a placeholder container
        player_avg_strategy = self.average_strategies_data[2]
        player_avg_strategy_sum = self.regrets_sum[2]

        player_avg_strategy_sum = torch.sum(player_avg_strategy, 0)
        player_avg_strategy.div_(player_avg_strategy_sum.expand_as(player_avg_strategy))

        # if the strategy is 'empty' (zero reach), strategy does not matter but we need to make sure
        # it sums to one -> now we set to always fold
        player_avg_strategy[0][torch.ne(player_avg_strategy[0], player_avg_strategy[0]).bool()] = 1
        player_avg_strategy[torch.ne(player_avg_strategy, player_avg_strategy).bool()] = 0

    # --- Normalizes the players' average counterfactual values.
    # --
    # -- Used at the end of re-solving so that we can track un-normalized average
    # -- cfvs, which are simpler to compute.
    # -- @local
    def _compute_normalize_average_cfvs(self):
        self.average_cfvs_data[1].div_(arguments.cfr_iters - arguments.cfr_skip_iters)
