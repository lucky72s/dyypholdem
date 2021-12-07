
import settings.arguments as arguments
import settings.game_settings as game_settings
import settings.constants as constants

from tree.tree_node import TreeNode
from nn.value_nn import ValueNn
from nn.next_round_value import NextRoundValue
from nn.next_round_value_pre import NextRoundValuePre

# used to load NNs and next_round_value_pre only once
neural_net = dict()
aux_net = None
next_round_pre = None


class LookaheadBuilder(object):

    def __init__(self, lookahead):
        self.lookahead = lookahead
        self.ccall_action_index = 1
        self.fold_action_index = 2

    # --- Builds the lookahead's internal data structures using the public tree.
    # -- @param tree the public tree used to construct the lookahead
    def build_from_tree(self, tree: TreeNode):

        self.lookahead.tree = tree
        self.lookahead.depth = tree.depth

        # per layer information about tree actions
        # per layer actions are the max number of actions for any of the nodes on the layer
        self.lookahead.bets_count = {}
        self.lookahead.nonallinbets_count = {}
        self.lookahead.terminal_actions_count = {}
        self.lookahead.actions_count = {}

        self.lookahead.first_call_terminal = self.lookahead.tree.children[1].terminal
        self.lookahead.first_call_transition = self.lookahead.tree.children[1].current_player == constants.Players.Chance
        self.lookahead.first_call_check = (not self.lookahead.first_call_terminal) and (not self.lookahead.first_call_transition)

        self._compute_tree_structures([tree], 1)
        # construct the initial data structures using the bet counts
        self._construct_data_structures()

        # action ids for first
        self.lookahead.parent_action_id = {}

        # traverse the tree and fill the datastructures (pot sizes, non-existing actions, ...)
        # node, layer, action, parent_action, gp_id
        self._set_datastructures_from_tree_dfs(tree, 1, 1, 1, 1, -100)

        # set additional info
        assert self.lookahead.terminal_actions_count[1] == 1 or self.lookahead.terminal_actions_count[1] == 2

        # we mask out fold as a possible action when check is for free, due to
        # 1) fewer actions means faster convergence
        # 2) we need to make sure prob of free fold is zero because ACPC dealer changes such action to check
        if self.lookahead.tree.bets[0] == self.lookahead.tree.bets[1]:
            self.lookahead.empty_action_mask[2][0].fill_(0)

        # construct the neural net query boxes
        self._construct_transition_boxes()

    def reset(self):
        for d in range(1, self.lookahead.depth + 1):
            if d in self.lookahead.ranges_data and self.lookahead.ranges_data[d] is not None:
                self.lookahead.ranges_data[d].fill_(1.0 / game_settings.hand_count)
            if d in self.lookahead.average_strategies_data and self.lookahead.average_strategies_data[d] is not None:
                self.lookahead.average_strategies_data[d].fill_(0)
            if d in self.lookahead.current_strategy_data and self.lookahead.current_strategy_data[d] is not None:
                self.lookahead.current_strategy_data[d].fill_(0)
            if d in self.lookahead.cfvs_data and self.lookahead.cfvs_data[d] is not None:
                self.lookahead.cfvs_data[d].fill_(0)
            if d in self.lookahead.average_cfvs_data and self.lookahead.average_cfvs_data[d] is not None:
                self.lookahead.average_cfvs_data[d].fill_(0)
            if d in self.lookahead.regrets_data and self.lookahead.regrets_data[d] is not None:
                self.lookahead.regrets_data[d].fill_(0)
            if d in self.lookahead.current_regrets_data and self.lookahead.current_regrets_data[d] is not None:
                self.lookahead.current_regrets_data[d].fill_(0)
            if d in self.lookahead.positive_regrets_data and self.lookahead.positive_regrets_data[d] is not None:
                self.lookahead.positive_regrets_data[d].fill_(0)
            if d in self.lookahead.placeholder_data and self.lookahead.placeholder_data[d] is not None:
                self.lookahead.placeholder_data[d].fill_(0)
            if d in self.lookahead.regrets_sum and self.lookahead.regrets_sum[d] is not None:
                self.lookahead.regrets_sum[d].fill_(0)
            if d in self.lookahead.inner_nodes and self.lookahead.inner_nodes[d] is not None:
                self.lookahead.inner_nodes[d].fill_(0)
            if d in self.lookahead.inner_nodes_p1 and self.lookahead.inner_nodes_p1[d] is not None:
                self.lookahead.inner_nodes_p1[d].fill_(0)
            if d in self.lookahead.swap_data and self.lookahead.swap_data[d] is not None:
                self.lookahead.swap_data[d].fill_(0)

        if self.lookahead.next_street_boxes is not None:
            self.lookahead.next_street_boxes.iter = 0
            self.lookahead.next_street_boxes.start_computation(self.lookahead.next_round_pot_sizes, self.lookahead.batch_size)

    # --- Computes the maximum number of actions at each depth of the tree.
    # --
    # -- Used to find the size for the tensors which store lookahead data. The
    # -- maximum number of actions is used so that every node at that depth can
    # -- fit in the same tensor.
    # -- @param current_layer a list of tree nodes at the current depth
    # -- @param current_depth the depth of the current tree nodes
    # -- @local
    def _compute_tree_structures(self, current_layer, current_depth):
        layer_actions_count = 0
        layer_terminal_actions_count = 0
        next_layer = []

        for n in range(0, len(current_layer)):
            node = current_layer[n]
            layer_actions_count = max(layer_actions_count, len(node.children))
            node_terminal_actions_count = 0
            for c in range(0, len(current_layer[n].children)):
                if node.children[c].terminal or node.children[c].current_player == constants.Players.Chance:
                    node_terminal_actions_count = node_terminal_actions_count + 1

            layer_terminal_actions_count = max(layer_terminal_actions_count, node_terminal_actions_count)
            # add children of the node to the next layer for later pass of BFS
            if not node.terminal:
                for c in range(0, len(node.children)):
                    next_layer.append(node.children[c])

        assert (layer_actions_count == 0) == (len(next_layer) == 0)
        assert (layer_actions_count == 0) == (current_depth == self.lookahead.depth)

        # set action and bet counts
        self.lookahead.bets_count[current_depth] = layer_actions_count - layer_terminal_actions_count
        self.lookahead.nonallinbets_count[current_depth] = layer_actions_count - layer_terminal_actions_count
        # remove allin
        self.lookahead.nonallinbets_count[current_depth] -= 1
        # if no alllin...
        if layer_actions_count == 2:
            assert layer_actions_count == layer_terminal_actions_count, "error in tree"
            self.lookahead.nonallinbets_count[current_depth] = 0
        self.lookahead.terminal_actions_count[current_depth] = layer_terminal_actions_count
        self.lookahead.actions_count[current_depth] = layer_actions_count

        if len(next_layer) > 0:
            assert layer_actions_count >= 2, "error in tree"
            # go deeper
            self._compute_tree_structures(next_layer, current_depth + 1)

    # --- Builds the tensors that store lookahead data during re-solving.
    def _construct_data_structures(self):

        self._compute_structure()

        # lookahead main data structures
        # all the structures are per-layer tensors, that is, each layer holds the data in n-dimensional tensors
        self.lookahead.pot_size = {}
        self.lookahead.ranges_data = {}
        self.lookahead.average_strategies_data = {}
        self.lookahead.current_strategy_data = {}
        self.lookahead.cfvs_data = {}
        self.lookahead.average_cfvs_data = {}
        self.lookahead.regrets_data = {}
        self.lookahead.current_regrets_data = {}
        self.lookahead.positive_regrets_data = {}
        self.lookahead.placeholder_data = {}
        self.lookahead.regrets_sum = {}
        self.lookahead.empty_action_mask = {}  # --used to mask empty actions

        # used to hold and swap inner (nonterminal) nodes when doing some transpose operations
        self.lookahead.inner_nodes = {}
        self.lookahead.inner_nodes_p1 = {}
        self.lookahead.swap_data = {}

        # create the data structure for the first two layers
        # data structures [actions x parent_action x grandparent_id x batch x players x range]
        self.lookahead.ranges_data[1] = arguments.Tensor(1, 1, 1, self.lookahead.batch_size, constants.players_count, game_settings.hand_count).fill_(1.0 / game_settings.hand_count)
        self.lookahead.ranges_data[2] = arguments.Tensor(self.lookahead.actions_count[1], 1, 1, self.lookahead.batch_size, constants.players_count, game_settings.hand_count).fill_(1.0 / game_settings.hand_count)
        self.lookahead.pot_size[1] = self.lookahead.ranges_data[1].clone().fill_(0)
        self.lookahead.pot_size[2] = self.lookahead.ranges_data[2].clone().fill_(0)
        self.lookahead.cfvs_data[1] = self.lookahead.ranges_data[1].clone().fill_(0)
        self.lookahead.cfvs_data[2] = self.lookahead.ranges_data[2].clone().fill_(0)
        self.lookahead.average_cfvs_data[1] = self.lookahead.ranges_data[1].clone().fill_(0)
        self.lookahead.average_cfvs_data[2] = self.lookahead.ranges_data[2].clone().fill_(0)
        self.lookahead.placeholder_data[1] = self.lookahead.ranges_data[1].clone().fill_(0)
        self.lookahead.placeholder_data[2] = self.lookahead.ranges_data[2].clone().fill_(0)

        # data structures for one player [actions x parent_action x grandparent_id x batch x 1 x range]
        self.lookahead.average_strategies_data[1] = None
        self.lookahead.average_strategies_data[2] = arguments.Tensor(self.lookahead.actions_count[1], 1, 1, self.lookahead.batch_size, game_settings.hand_count).fill_(0)
        self.lookahead.current_strategy_data[1] = None
        self.lookahead.current_strategy_data[2] = self.lookahead.average_strategies_data[2].clone().fill_(0)
        self.lookahead.regrets_data[1] = None
        self.lookahead.regrets_data[2] = self.lookahead.average_strategies_data[2].clone().fill_(0)
        self.lookahead.current_regrets_data[1] = None
        self.lookahead.current_regrets_data[2] = self.lookahead.average_strategies_data[2].clone().fill_(0)
        self.lookahead.positive_regrets_data[1] = None
        self.lookahead.positive_regrets_data[2] = self.lookahead.average_strategies_data[2].clone().fill_(0)
        self.lookahead.empty_action_mask[1] = None
        self.lookahead.empty_action_mask[2] = self.lookahead.average_strategies_data[2].clone().fill_(1)

        # data structures for summing over the actions [1 x parent_action x grandparent_id x batch x range]
        self.lookahead.regrets_sum[1] = arguments.Tensor(1, 1, 1, self.lookahead.batch_size, game_settings.hand_count).fill_(0)
        self.lookahead.regrets_sum[2] = arguments.Tensor(1, self.lookahead.bets_count[1], 1, self.lookahead.batch_size, game_settings.hand_count).fill_(0)

        # data structures for inner nodes (not terminal nor allin) [bets_count x parent_nonallinbetscount x gp_id x batch x players x range]
        self.lookahead.inner_nodes[1] = arguments.Tensor(1, 1, 1, self.lookahead.batch_size, constants.players_count, game_settings.hand_count).fill_(0)
        self.lookahead.swap_data[1] = self.lookahead.inner_nodes[1].transpose(1, 2).clone()
        self.lookahead.inner_nodes_p1[1] = arguments.Tensor(1, 1, 1, self.lookahead.batch_size, 1, game_settings.hand_count).fill_(0)

        if self.lookahead.depth > 2:
            self.lookahead.inner_nodes[2] = arguments.Tensor(self.lookahead.bets_count[1], 1, 1, self.lookahead.batch_size, constants.players_count, game_settings.hand_count).fill_(0)
            self.lookahead.swap_data[2] = self.lookahead.inner_nodes[2].transpose(1, 2).clone()
            self.lookahead.inner_nodes_p1[2] = arguments.Tensor(self.lookahead.bets_count[1], 1, 1, self.lookahead.batch_size, 1, game_settings.hand_count).fill_(0)

        # create the data structures for the rest of the layers
        for d in range(3, self.lookahead.depth + 1):
            # data structures [actions x parent_action x grandparent_id x batch x players x range]
            self.lookahead.ranges_data[d] = arguments.Tensor(self.lookahead.actions_count[d - 1], self.lookahead.bets_count[d - 2], self.lookahead.nonterminal_nonallin_nodes_count[d - 2], self.lookahead.batch_size, constants.players_count,
                                                             game_settings.hand_count).fill_(0)
            self.lookahead.cfvs_data[d] = self.lookahead.ranges_data[d].clone()
            self.lookahead.placeholder_data[d] = self.lookahead.ranges_data[d].clone()
            self.lookahead.pot_size[d] = self.lookahead.ranges_data[d].clone().fill_(game_settings.stack)

            # data structures [actions x parent_action x grandparent_id x batch x 1 x range]
            self.lookahead.average_strategies_data[d] = arguments.Tensor(self.lookahead.actions_count[d - 1], self.lookahead.bets_count[d - 2], self.lookahead.nonterminal_nonallin_nodes_count[d - 2], self.lookahead.batch_size,
                                                                         game_settings.hand_count).fill_(0)
            self.lookahead.current_strategy_data[d] = self.lookahead.average_strategies_data[d].clone()
            self.lookahead.regrets_data[d] = self.lookahead.average_strategies_data[d].clone().fill_(self.lookahead.regret_epsilon)
            self.lookahead.current_regrets_data[d] = self.lookahead.average_strategies_data[d].clone().fill_(0)
            self.lookahead.empty_action_mask[d] = self.lookahead.average_strategies_data[d].clone().fill_(1)
            self.lookahead.positive_regrets_data[d] = self.lookahead.regrets_data[d].clone()

            # data structures [1 x parent_action x grandparent_id x batch x players x range]
            self.lookahead.regrets_sum[d] = arguments.Tensor(1, self.lookahead.bets_count[d - 2], self.lookahead.nonterminal_nonallin_nodes_count[d - 2], self.lookahead.batch_size, constants.players_count, game_settings.hand_count).fill_(0)

            # data structures for the layers except the last one
            if d < self.lookahead.depth:
                self.lookahead.inner_nodes[d] = arguments.Tensor(self.lookahead.bets_count[d - 1], self.lookahead.nonallinbets_count[d - 2], self.lookahead.nonterminal_nonallin_nodes_count[d - 2], self.lookahead.batch_size,
                                                                 constants.players_count, game_settings.hand_count).fill_(0)
                self.lookahead.inner_nodes_p1[d] = arguments.Tensor(self.lookahead.bets_count[d - 1], self.lookahead.nonallinbets_count[d - 2], self.lookahead.nonterminal_nonallin_nodes_count[d - 2], self.lookahead.batch_size, 1,
                                                                    game_settings.hand_count).fill_(0)
                self.lookahead.swap_data[d] = self.lookahead.inner_nodes[d].transpose(1, 2).clone()

        # create the optimized data structures for terminal equity
        self.lookahead.term_call_indices = {}
        self.lookahead.num_term_call_nodes = 0
        self.lookahead.term_fold_indices = {}
        self.lookahead.num_term_fold_nodes = 0

        # calculate term_call_indices
        for d in range(2, self.lookahead.depth + 1):
            if self.lookahead.tree.street != constants.streets_count:
                if d > 2 or self.lookahead.first_call_terminal:
                    before = self.lookahead.num_term_call_nodes
                    self.lookahead.num_term_call_nodes = self.lookahead.num_term_call_nodes + self.lookahead.ranges_data[d][1][-1].size(0)
                    self.lookahead.term_call_indices[d] = [before, self.lookahead.num_term_call_nodes]
            else:
                if d > 2 or self.lookahead.first_call_terminal:
                    before = self.lookahead.num_term_call_nodes
                    self.lookahead.num_term_call_nodes = self.lookahead.num_term_call_nodes + self.lookahead.ranges_data[d][1].size(0) * self.lookahead.ranges_data[d][1].size(1)
                    self.lookahead.term_call_indices[d] = [before, self.lookahead.num_term_call_nodes]

        # calculate term_fold_indices
        for d in range(2, self.lookahead.depth + 1):
            before = self.lookahead.num_term_fold_nodes
            self.lookahead.num_term_fold_nodes = self.lookahead.num_term_fold_nodes + self.lookahead.ranges_data[d][0].size(0) * self.lookahead.ranges_data[d][0].size(1)
            self.lookahead.term_fold_indices[d] = [before, self.lookahead.num_term_fold_nodes]

        self.lookahead.ranges_data_call = arguments.Tensor(self.lookahead.num_term_call_nodes, self.lookahead.batch_size, constants.players_count, game_settings.hand_count)
        self.lookahead.ranges_data_fold = arguments.Tensor(self.lookahead.num_term_fold_nodes, self.lookahead.batch_size, constants.players_count, game_settings.hand_count)

        self.lookahead.cfvs_data_call = arguments.Tensor(self.lookahead.num_term_call_nodes, self.lookahead.batch_size, constants.players_count, game_settings.hand_count)
        self.lookahead.cfvs_data_fold = arguments.Tensor(self.lookahead.num_term_fold_nodes, self.lookahead.batch_size, constants.players_count, game_settings.hand_count)

    # --- Computes the number of nodes at each depth of the tree.
    # --
    # -- Used to find the size for the tensors which store lookahead data.
    # -- @local
    def _compute_structure(self):
        assert 1 <= self.lookahead.tree.street <= constants.streets_count

        self.lookahead.regret_epsilon = 1.0 / 1000000000

        # which player acts at particular depth
        # self.lookahead.acting_player = arguments.Tensor(self.lookahead.depth + 1).fill_(-1)
        self.lookahead.acting_player = {}
        self.lookahead.acting_player[1] = 1  # in lookahead, 1 does not stand for player IDs, it's just the first player to act
        for d in range(2, self.lookahead.depth + 2):
            self.lookahead.acting_player[d] = 3 - self.lookahead.acting_player[d - 1]

        self.lookahead.bets_count[-1] = 1
        self.lookahead.bets_count[0] = 1
        self.lookahead.nonallinbets_count[-1] = 1
        self.lookahead.nonallinbets_count[0] = 1
        self.lookahead.terminal_actions_count[-1] = 0
        self.lookahead.terminal_actions_count[0] = 0
        self.lookahead.actions_count[-1] = 1
        self.lookahead.actions_count[0] = 1

        # compute the node counts
        self.lookahead.nonterminal_nodes_count = {}
        self.lookahead.nonterminal_nonallin_nodes_count = {}
        self.lookahead.all_nodes_count = {}
        self.lookahead.allin_nodes_count = {}
        self.lookahead.inner_nodes_count = {}

        self.lookahead.nonterminal_nodes_count[1] = 1
        self.lookahead.nonterminal_nodes_count[2] = self.lookahead.bets_count[1]
        self.lookahead.nonterminal_nonallin_nodes_count[0] = 1
        self.lookahead.nonterminal_nonallin_nodes_count[1] = 1
        self.lookahead.nonterminal_nonallin_nodes_count[2] = self.lookahead.nonterminal_nodes_count[2]
        self.lookahead.nonterminal_nonallin_nodes_count[2] = self.lookahead.nonterminal_nonallin_nodes_count[2] - 1
        self.lookahead.all_nodes_count[1] = 1
        self.lookahead.all_nodes_count[2] = self.lookahead.actions_count[1]
        self.lookahead.allin_nodes_count[1] = 0
        self.lookahead.allin_nodes_count[2] = 1
        self.lookahead.inner_nodes_count[1] = 1
        self.lookahead.inner_nodes_count[2] = 1

        for d in range(2, self.lookahead.depth):
            self.lookahead.all_nodes_count[d + 1] = self.lookahead.nonterminal_nonallin_nodes_count[d - 1] * self.lookahead.bets_count[d - 1] * self.lookahead.actions_count[d]
            self.lookahead.allin_nodes_count[d + 1] = self.lookahead.nonterminal_nonallin_nodes_count[d - 1] * self.lookahead.bets_count[d - 1] * 1
            self.lookahead.nonterminal_nodes_count[d + 1] = self.lookahead.nonterminal_nonallin_nodes_count[d - 1] * self.lookahead.nonallinbets_count[d - 1] * self.lookahead.bets_count[d]
            self.lookahead.nonterminal_nonallin_nodes_count[d + 1] = self.lookahead.nonterminal_nonallin_nodes_count[d - 1] * self.lookahead.nonallinbets_count[d - 1] * self.lookahead.nonallinbets_count[d]

    # --- Traverses the tree to fill in lookahead data structures that summarize data
    # -- contained in the tree.
    # --
    # -- For example, saves pot sizes and numbers of actions at each lookahead state.
    # --
    # -- @param node the current node of the public tree
    # -- @param layer the depth of the current node
    # -- @param action_id the index of the action that led to this node
    # -- @param parent_id the index of the current node's parent
    # -- @param gp_id the index of the current node's grandparent
    # -- @local
    def _set_datastructures_from_tree_dfs(self, node, layer, action_id, parent_id, gp_id, cur_action_id, parent_action_id=None):
        assert node.pot, "Node has no pot"

        self.lookahead.pot_size[layer][action_id - 1:action_id, parent_id - 1:parent_id, gp_id - 1:gp_id, ...].fill_(node.pot)

        if layer == 3 and cur_action_id == constants.Actions.ccall.value:
            self.lookahead.parent_action_id[parent_id] = parent_action_id

        node.lookahead_coordinates = arguments.Tensor([action_id, parent_id, gp_id])

        # transition call cannot be allin call
        if node.current_player == constants.Players.Chance:
            assert parent_id <= self.lookahead.nonallinbets_count[layer - 2]

        if layer < self.lookahead.depth + 1:
            gp_nonallinbets_count = self.lookahead.nonallinbets_count[layer - 2]
            prev_layer_terminal_actions_count = self.lookahead.terminal_actions_count[layer - 1]
            gp_terminal_actions_count = self.lookahead.terminal_actions_count[layer - 2]
            prev_layer_bets_count = self.lookahead.bets_count[layer - 1]

            # compute next coordinates for parent and grandparent
            next_parent_id = action_id - prev_layer_terminal_actions_count
            next_gp_id = (gp_id - 1) * gp_nonallinbets_count + parent_id

            if (not node.terminal) and (node.current_player != constants.Players.Chance):
                # parent is not an allin raise
                assert parent_id <= self.lookahead.nonallinbets_count[layer - 2]

                # do we need to mask some actions for that node? (that is, does the node have fewer children than the max number of children for any node on this layer)
                node_with_empty_actions = (len(node.children) < self.lookahead.actions_count[layer])

                if node_with_empty_actions:
                    # we need to mask non-existing padded bets
                    assert layer > 1
                    terminal_actions_count = self.lookahead.terminal_actions_count[layer]
                    assert (terminal_actions_count == 2)
                    existing_bets_count = len(node.children) - terminal_actions_count

                    # allin situations
                    if existing_bets_count == 0:
                        assert action_id == self.lookahead.actions_count[layer - 1]

                    for child_id in range(0, terminal_actions_count):
                        child_node = node.children[child_id]
                        # go deeper
                        self._set_datastructures_from_tree_dfs(child_node, layer + 1, child_id + 1, next_parent_id, next_gp_id, node.actions[child_id], cur_action_id)

                    # we need to make sure that even though there are fewer actions, the last action/allin is has the same last index as if we had full number of actions
                    # we manually set the action_id as the last action (allin)
                    for b in range(0, existing_bets_count):
                        self._set_datastructures_from_tree_dfs(node.children[len(node.children) - b - 1], layer + 1, self.lookahead.actions_count[layer] - b,
                                                               next_parent_id, next_gp_id, node.actions[len(node.children) - b - 1], cur_action_id)

                    # mask out empty actions
                    # self.lookahead.empty_action_mask[layer+1][terminal_actions_count:terminal_actions_count-(existing_bets_count+1), next_parent_id-1:1, next_gp_id-1:1, :] = 0
                    self.lookahead.empty_action_mask[layer + 1].narrow(0, terminal_actions_count, self.lookahead.empty_action_mask[layer + 1].size(0) -
                                                                       existing_bets_count - terminal_actions_count).narrow(1, next_parent_id - 1, 1).narrow(2, next_gp_id - 1, 1).fill_(0)
                else:
                    # node has full action count, easy to handle
                    for child_id in range(0, len(node.children)):
                        child_node = node.children[child_id]
                        # go deeper
                        self._set_datastructures_from_tree_dfs(child_node, layer + 1, child_id + 1, next_parent_id, next_gp_id, node.actions[child_id], cur_action_id)

    # --- Builds the neural net query boxes which estimate counterfactual values
    # -- at depth-limited states of the lookahead.
    # -- @local
    def _construct_transition_boxes(self):
        global neural_net
        global aux_net
        global next_round_pre

        # nothing to do if at the river
        if self.lookahead.tree.street == constants.streets_count:
            return

        # load neural nets if not already loaded
        nn = neural_net.get(self.lookahead.tree.street) or ValueNn().load_for_street(self.lookahead.tree.street)
        neural_net[self.lookahead.tree.street] = nn
        if self.lookahead.tree.street == 1:
            aux_net = aux_net or ValueNn().load_for_street(self.lookahead.tree.street, True)

        self.lookahead.next_street_boxes = None
        self.lookahead.next_street_boxes_aux = None
        self.lookahead.indices = {}
        self.lookahead.num_pot_sizes = 0

        if self.lookahead.tree.street == 1:
            self.lookahead.next_street_boxes = next_round_pre or NextRoundValuePre(nn, aux_net, self.lookahead.terminal_equity.board)
            next_round_pre = self.lookahead.next_street_boxes
        else:
            self.lookahead.next_street_boxes = NextRoundValue(nn, self.lookahead.terminal_equity.board)

        # create the optimized data structures for batching next_round_value
        for d in range(2, self.lookahead.depth + 1):
            if d == 2 and self.lookahead.first_call_transition:
                before = self.lookahead.num_pot_sizes
                self.lookahead.num_pot_sizes = self.lookahead.num_pot_sizes + 1
                self.lookahead.indices[d] = [before, self.lookahead.num_pot_sizes]
            # elif not game_settings.nl and (d > 2 or self.lookahead.first_call_transition):
            #     before = self.lookahead.num_pot_sizes
            #     self.lookahead.num_pot_sizes = self.lookahead.num_pot_sizes + (self.lookahead.pot_size[d][1].size(0)) * self.lookahead.pot_size[d][1].size(1)
            #     self.lookahead.indices[d] = [before, self.lookahead.num_pot_sizes]
            elif self.lookahead.pot_size[d][1].size(0) > 1:
                before = self.lookahead.num_pot_sizes
                self.lookahead.num_pot_sizes = self.lookahead.num_pot_sizes + (self.lookahead.pot_size[d][1].size(0) - 1) * self.lookahead.pot_size[d][1].size(1)
                self.lookahead.indices[d] = [before, self.lookahead.num_pot_sizes]

        if self.lookahead.num_pot_sizes == 0:
            return

        self.lookahead.next_round_pot_sizes = arguments.Tensor(self.lookahead.num_pot_sizes).zero_()

        self.lookahead.action_to_index = {}
        for d in range(2, self.lookahead.depth + 1):
            parent_indices = [0, self.lookahead.pot_size[d].size(1) - 1]
            if self.lookahead.indices.get(d):
                if d == 2:
                    parent_indices = [0, 1]
                destination = self.lookahead.next_round_pot_sizes[self.lookahead.indices[d][0]:self.lookahead.indices[d][1]]
                source = self.lookahead.pot_size[d][1, parent_indices[0]:parent_indices[1], :, 0, 0, 0].view(destination.shape)
                destination.copy_(source)
                if d <= 3:
                    if d == 2:
                        assert self.lookahead.indices[d][0] == self.lookahead.indices[d][1] - 1  # TODO: check if this is correct ?!?!?
                        self.lookahead.action_to_index[constants.Actions.ccall.value] = self.lookahead.indices[d][0]
                    else:
                        assert self.lookahead.pot_size[d][1, parent_indices[0]:parent_indices[1]].size(1) == 1, 'bad num_indices: '
                        for parent_action_idx in range(1, self.lookahead.pot_size[d][1].size(0) + 1):
                            action_id = self.lookahead.parent_action_id[parent_action_idx]
                            assert self.lookahead.action_to_index.get(action_id) is None
                            self.lookahead.action_to_index[action_id.item()] = self.lookahead.indices[d][0] + parent_action_idx - 1

        if self.lookahead.action_to_index.get(constants.Actions.ccall.value) is None:
            print(self.lookahead.action_to_index)
            print(self.lookahead.parent_action_id)
            assert False, "this should not happen"

        self.lookahead.next_street_boxes.start_computation(self.lookahead.next_round_pot_sizes, self.lookahead.batch_size)
        self.lookahead.next_street_boxes_inputs = arguments.Tensor(self.lookahead.num_pot_sizes, self.lookahead.batch_size, constants.players_count, game_settings.hand_count).zero_()
        self.lookahead.next_street_boxes_outputs = self.lookahead.next_street_boxes_inputs.clone()
