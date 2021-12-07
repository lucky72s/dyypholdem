
import settings.arguments as arguments

from terminal_equity.terminal_equity import TerminalEquity
from tree.tree_builder import PokerTreeBuilder
from tree.tree_node import TreeNode, BuildTreeParams
from lookahead.lookahead import Lookahead
from lookahead.resolve_results import ResolveResults
import game.card_tools as card_tools


class Resolving(object):

    tree_builder: PokerTreeBuilder
    terminal_equity: TerminalEquity
    player_range: arguments.Tensor
    opponent_range: arguments.Tensor
    opponent_cfvs: object
    lookahead_tree: TreeNode
    lookahead: Lookahead
    resolve_results: ResolveResults

    def __init__(self, terminal_equity):
        self.tree_builder = PokerTreeBuilder()
        self.terminal_equity = terminal_equity

    # --- Builds a depth-limited public tree rooted at a given game node.
    # -- @param node the root of the tree
    # -- @local
    def _create_lookahead_tree(self, node):
        build_tree_params = BuildTreeParams(root_node=node, limit_to_street=True)
        self.lookahead_tree = self.tree_builder.build_tree(build_tree_params)

    # -- Re-solves a depth-limited lookahead using input ranges.
    # --
    # -- Uses the input range for the opponent instead of a gadget range, so only
    # -- appropriate for re-solving the root node of the game tree (where ranges
    # -- are fixed).
    # --
    # -- @param node the public node at which to re-solve
    # -- @param player_range a range vector for the re-solving player
    # -- @param opponent_range a range vector for the opponent
    def resolve_first_node(self, node, player_range, opponent_range) -> ResolveResults:

        arguments.logger.debug(f"Resolving first node with {arguments.cfr_iters} iterations")

        self.player_range = player_range
        self.opponent_range = opponent_range
        self.opponent_cfvs = None

        self._create_lookahead_tree(node)

        if player_range.dim() == 1:
            player_range = player_range.view(1, player_range.size(0))
            opponent_range = opponent_range.view(1, opponent_range.size(0))

        self.lookahead = Lookahead(self.terminal_equity, player_range.size(0))

        arguments.timer.split_start("Building lookahead tree", log_level="TRACE")
        self.lookahead.build_lookahead(self.lookahead_tree)
        arguments.timer.split_stop("Lookahead tree build time", log_level="TIMING")

        arguments.timer.split_start(f"Resolving tree", log_level="TRACE")
        self.lookahead.resolve_first_node(player_range, opponent_range)
        arguments.timer.split_stop("Tree resolution time", log_level="TIMING")

        self.resolve_results = self.lookahead.get_results()

        return self.resolve_results

    # -- Re-solves a depth-limited lookahead using an input range for the player and
    # -- the @{cfrd_gadget|CFRDGadget} to generate ranges for the opponent.
    # --
    # -- @param node the public node at which to re-solve
    # -- @param player_range a range vector for the re-solving player
    # -- @param opponent_cfvs a vector of cfvs achieved by the opponent
    # -- before re-solving
    def resolve(self, node, player_range, opponent_cfvs):
        assert card_tools.is_valid_range(player_range, node.board)

        self.player_range = player_range
        self.opponent_cfvs = opponent_cfvs
        self._create_lookahead_tree(node)

        if player_range.dim() == 1:
            player_range = player_range.view(1, player_range.size(0))

        arguments.timer.split_start("Building lookahead tree", log_level="TRACE")
        self.lookahead = Lookahead(self.terminal_equity, player_range.size(0))
        self.lookahead.build_lookahead(self.lookahead_tree)
        arguments.timer.split_stop("Tree build time", log_level="TIMING")

        arguments.timer.split_start("Resolving node", log_level="TRACE")
        self.lookahead.resolve(player_range, opponent_cfvs)
        arguments.timer.split_stop("Resolve time", log_level="TIMING")

        self.resolve_results = self.lookahead.get_results()
        return self.resolve_results

    # --- Gives a list of possible actions at the node being re-solved.
    # --
    # -- The node must first be re-solved with @{resolve} or @{resolve_first_node}.
    # -- @return a list of legal actions
    def get_possible_actions(self):
        return self.lookahead_tree.actions

    # --- Gives the average counterfactual values that the re-solve player received
    # -- at the node during re-solving.
    # --
    # -- The node must first be re-solved with @{resolve_first_node}.
    # --
    # -- @return a vector of cfvs
    def get_root_cfv(self):
        return self.resolve_results.root_cfvs

    # --- Gives the average counterfactual values that each player received
    # -- at the node during re-solving.
    # --
    # -- Useful for data generation for neural net training
    # --
    # -- The node must first be re-solved with @{resolve_first_node}.
    # --
    # -- @return a 2xK tensor of cfvs, where K is the range size
    def get_root_cfv_both_players(self):
        return self.resolve_results.root_cfvs_both_players

    # --- Gives the average counterfactual values that the opponent received
    # -- during re-solving after the re-solve player took a given action.
    # --
    # -- Used during continual re-solving to track opponent cfvs. The node must
    # -- first be re-solved with @{resolve} or @{resolve_first_node}.
    # --
    # -- @param action the action taken by the re-solve player at the node being
    # -- re-solved
    # -- @return a vector of cfvs
    def get_action_cfv(self, action):
        action_id = self._action_to_action_id(action)
        return self.resolve_results.children_cfvs[action_id]

    # --- Gives the average counterfactual values that the opponent received
    # -- during re-solving after a chance event (the betting round changes and
    # -- more cards are dealt).
    # --
    # -- Used during continual re-solving to track opponent cfvs. The node must
    # -- first be re-solved with @{resolve} or @{resolve_first_node}.
    # --
    # -- @param action the action taken by the re-solve player at the node being
    # -- re-solved
    # -- @param board a vector of board cards which were updated by the chance event
    # -- @return a vector of cfvs
    def get_chance_action_cfv(self, action, board):
        # resolve to get next_board chance actions if flop
        if board.dim() == 1 and board.size(0) == 3:
            self.lookahead.reset()
            board_idx = card_tools.get_flop_board_index(board)
            self.lookahead.next_board_idx = board_idx

            if self.opponent_cfvs is not None:
                self.lookahead.resolve(self.player_range, self.opponent_cfvs)
            else:
                self.lookahead.resolve_first_node(self.player_range, self.opponent_range)
            self.lookahead.next_board_idx = None
        return self.lookahead.get_chance_action_cfv(action, board)

    # --- Gives the probability that the re-solved strategy takes a given action.
    # --
    # -- The node must first be re-solved with @{resolve} or @{resolve_first_node}.
    # --
    # -- @param action a legal action at the re-solve node
    # -- @return a vector giving the probability of taking the action with each
    # -- private hand
    def get_action_strategy(self, action):
        action_id = self._action_to_action_id(action)
        return self.resolve_results.strategy[action_id][0]

    # --- Gives the index of the given action at the node being re-solved.
    # --
    # -- The node must first be re-solved with @{resolve} or @{resolve_first_node}.
    # -- @param action a legal action at the node
    # -- @return the index of the action
    # -- @local
    def _action_to_action_id(self, action):
        actions = self.get_possible_actions()
        action_id = -1
        for i in range(actions.size(0)):
            if action == actions[i]:
                action_id = i
        assert action_id != -1
        return action_id
