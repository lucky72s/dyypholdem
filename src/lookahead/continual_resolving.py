
import torch

import settings.arguments as arguments
import settings.constants as constants
import settings.game_settings as game_settings

from server.protocol_to_node import ProcessedState, Action
import game.card_tools as card_tools
from terminal_equity.terminal_equity import TerminalEquity
from lookahead.resolving import Resolving
from tree.tree_node import TreeNode

import utils.pseudo_random as random_


class ContinualResolving(object):

    last_node: TreeNode
    decision_id: int
    last_bet: int
    position: int
    player: constants.Players
    hand_id: int

    def __init__(self):
        self.starting_player_range = card_tools.get_uniform_range(arguments.Tensor())
        self.terminal_equity = TerminalEquity()
        self.first_node_resolving: Resolving = None
        self.starting_cfvs_p1: arguments.Tensor = None

        self.resolve_first_node()

    # --- Solves a depth-limited lookahead from the first node of the game to get
    # -- opponent counterfactual values.
    # --
    # -- The cfvs are stored in the field `starting_cfvs_p1`. Because this is the
    # -- first node of the game, exact ranges are known for both players, so
    # -- opponent cfvs are not necessary for solving.
    def resolve_first_node(self):
        first_node = TreeNode()
        first_node.board = arguments.Tensor()
        first_node.street = 1
        first_node.current_player = constants.Players.P1
        first_node.bets = arguments.Tensor([game_settings.small_blind, game_settings.big_blind])
        first_node.num_bets = 1

        self.terminal_equity.set_board(first_node.board)
        # create the starting ranges
        player_range = card_tools.get_uniform_range(first_node.board)
        opponent_range = card_tools.get_uniform_range(first_node.board)

        # create re-solving and re-solve the first node
        self.first_node_resolving = Resolving(self.terminal_equity)
        self.first_node_resolving.resolve_first_node(first_node, player_range, opponent_range)
        # store the initial CFVs
        self.starting_cfvs_p1 = self.first_node_resolving.get_root_cfv()

    # --- Re-initializes the continual re-solving to start a new game from the root
    # -- of the game tree.
    # -- @param state the first state where the re-solving player acts in the new
    # -- game (a table of the type returned by @{protocol_to_node.parse_state})
    def start_new_hand(self, state: ProcessedState):
        self.last_node = None
        self.decision_id = 0
        self.position = state.position
        self.player = state.player
        self.hand_id = state.hand_id

    # --- Re-solves a node and chooses the re-solving player's next action.
    # -- @param node the game node where the re-solving player is to act (a table of
    # -- the type returned by @{protocol_to_node.parsed_state_to_node})
    # -- @param state the game state where the re-solving player is to act
    # -- (a table of the type returned by @{protocol_to_node.parse_state})
    # -- @return an action sampled from the re-solved strategy at the given state,
    # -- with the fields:
    # -- * `action`: an element of @{constants.acpc_actions}
    # -- * `raise_amount`: the number of chips to raise (if `action` is raise)
    def compute_action(self, state: ProcessedState, node: TreeNode):
        self._resolve_node(state, node)
        sampled_bet = self._sample_bet(state, node)

        self.decision_id = self.decision_id + 1
        self.last_bet = sampled_bet
        self.last_node = node

        out = self._bet_to_action(node, sampled_bet)

        return out

    # --- Re-solves a node to choose the re-solving player's next action.
    # -- @param node the game node where the re-solving player is to act (a table of
    # -- the type returned by @{protocol_to_node.parsed_state_to_node})
    # -- @param state the game state where the re-solving player is to act
    # -- (a table of the type returned by @{protocol_to_node.parse_state})
    # -- @local
    def _resolve_node(self, state, node):
        # 1.0 first node and P1 position
        # no need to update an invariant since this is the very first situation
        if self.decision_id == 0 and self.player == constants.Players.P1:
            # the strategy computation for the first decision node has been already set up
            self.current_player_range = self.starting_player_range.clone()
            self.resolving = self.first_node_resolving

        # 2.0 other nodes - we need to update the invariant
        else:
            assert not node.terminal
            assert node.current_player == self.player

            arguments.logger.debug(f"Resolving current node with {arguments.cfr_iters} iterations")

            # 2.1 update the invariant based on actions we did not make
            self._update_invariant(state, node)

            arguments.timer.start()
            arguments.timer.split_start("Calculating terminal equity...", log_level="TRACE")
            # 2.2 re-solve
            self.terminal_equity.set_board(node.board)
            arguments.timer.split_stop("Terminal equities time", log_level="TIMING")

            self.resolving = Resolving(self.terminal_equity)
            self.resolving.resolve(node, self.current_player_range, self.current_opponent_cfvs_bound)

            arguments.timer.stop("Node resolved and equities for actions calculated", log_level="DEBUG")

    # --- Updates the player's range and the opponent's counterfactual values to be
    # -- consistent with game actions since the last re-solved state.
    # -- Updates it only for actions we did not make, since we update the invariant for our action as soon as we make it.
    # --
    # -- @param node the game node where the re-solving player is to act (a table of
    # -- the type returned by @{protocol_to_node.parsed_state_to_node})
    # -- @param state the game state where the re-solving player is to act
    # -- (a table of the type returned by @{protocol_to_node.parse_state})
    # -- @local
    def _update_invariant(self, state, node):
        # 1.0 street has changed
        if self.last_node and self.last_node.street != node.street:
            assert self.last_node.street + 1 == node.street

            # 1.1 opponent cfvs
            # if the street has changed, the reconstruction API simply gives us CFVs
            self.current_opponent_cfvs_bound = self.resolving.get_chance_action_cfv(self.last_bet, node.board)
            # 1.2 player range
            # if street has change, we have to mask out the colliding hands
            self.current_player_range = card_tools.normalize_range(node.board, self.current_player_range)

        # 2.0 first decision for P2
        elif self.decision_id == 0:
            assert self.player == constants.Players.P2
            assert node.street == 1

            self.current_player_range = self.starting_player_range.clone()
            self.current_opponent_cfvs_bound = self.starting_cfvs_p1.clone()

        # 3.0 handle game within the street
        else:
            assert self.last_node.street == node.street

    # --- Samples an action to take from the strategy at the given game state.
    # -- @param node the game node where the re-solving player is to act (a table of
    # -- the type returned by @{protocol_to_node.parsed_state_to_node})
    # -- @param state the game state where the re-solving player is to act
    # -- (a table of the type returned by @{protocol_to_node.parse_state})
    # -- @return an index representing the action chosen
    # -- @local
    def _sample_bet(self, state, node):
        # 1.0 get the possible bets in the node
        possible_bets = self.resolving.get_possible_actions()
        actions_count = possible_bets.size(0)

        # 2.0 get the strategy for the current hand since the strategy is computed for all hands
        str_strategies = f"Strategy -> "
        hand_strategy = arguments.Tensor(actions_count)
        for i in range(actions_count):
            action_bet = possible_bets[i].item()
            action_strategy = self.resolving.get_action_strategy(action_bet)
            hand_strategy[i] = action_strategy[self.hand_id]
            if action_bet == -2:
                action_str = "Fold"
            elif action_bet == -1:
                action_str = "\tCheck/Call"
            elif action_bet == game_settings.stack:
                action_str = "\tAll-In"
            else:
                action_str = f"\tRaise {int(action_bet)}"
            action_str += f": {hand_strategy[i]:.5f}"
            str_strategies += action_str
        assert abs(1 - hand_strategy.sum()) < 0.001
        arguments.logger.success(str_strategies)

        # 3.0 sample the action by doing cumsum and uniform sample
        if arguments.use_pseudo_random:
            r = 0.55
        else:
            r = random_.random()

        hand_strategy_cumsum = torch.cumsum(hand_strategy, 0)
        sampled_bet = int(possible_bets[hand_strategy_cumsum.gt(r)][0].item())
        # change fold to check if it is free
        if sampled_bet == -2 and state.bet1 == state.bet2:
            sampled_bet = -1
            str_no_fold = "Changed FOLD action to free (!) CHECK"
            arguments.logger.warning(str_no_fold)

        if sampled_bet == -2:
            sampled_bet_action = "FOLD"
        elif sampled_bet == -1:
            if state.bet1 == state.bet2:
                sampled_bet_action = "CHECK"
            else:
                sampled_bet_action = "CALL"
        elif sampled_bet == game_settings.stack:
            sampled_bet_action = "ALL-IN"
        else:
            sampled_bet_action = "RAISE"
        lower_range_actions = hand_strategy_cumsum[hand_strategy_cumsum.lt(r)].tolist()
        higher_range_actions = hand_strategy_cumsum[hand_strategy_cumsum.gt(r)].tolist()
        str_action = f"Cumulated action cutoff {r:.3f} -> playing action in probability range {(lower_range_actions or [0])[-1]:.3f} to {higher_range_actions[0]:.3f} => {sampled_bet_action}"
        arguments.logger.success(str_action)

        # 4.0 update the invariants based on our action
        self.current_opponent_cfvs_bound = self.resolving.get_action_cfv(sampled_bet)
        strategy = self.resolving.get_action_strategy(sampled_bet)
        self.current_player_range.mul_(strategy)
        self.current_player_range = card_tools.normalize_range(node.board, self.current_player_range)

        return sampled_bet

    # --- Converts an internal action representation into a cleaner format.
    # -- @param node the game node where the re-solving player is to act (a table of
    # -- the type returned by @{protocol_to_node.parsed_state_to_node})
    # -- @param sampled_bet the index of the action to convert
    # -- @return a table specifying the action, with the fields:
    # --
    # -- * `action`: an element of @{constants.acpc_actions}
    # --
    # -- * `raise_amount`: the number of chips to raise (if `action` is raise)
    # -- @local
    @staticmethod
    def _bet_to_action(node, sampled_bet):
        if sampled_bet == constants.Actions.fold.value:
            return Action(action=constants.ACPCActions.fold)
        elif sampled_bet == constants.Actions.ccall.value:
            return Action(action=constants.ACPCActions.ccall)
        else:
            assert sampled_bet >= 0
            return Action(action=constants.ACPCActions.rraise, raise_amount=sampled_bet)

