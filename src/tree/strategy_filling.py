
import settings.arguments as arguments
import settings.constants as constants
import settings.game_settings as game_settings

import game.card_tools as card_tools


class StrategyFilling(object):

    def __init__(self):
        pass

    # --- Fills a public tree with a uniform strategy.
    # -- @param tree a public tree for Leduc Hold'em or variant
    def fill_uniform(self, tree):
        self._fill_uniform_dfs(tree)

    # --- Fills a node with a uniform strategy and recurses on the children.
    # -- @param node the node
    # -- @local
    def _fill_uniform_dfs(self, node):
        if node.current_player == constants.Players.Chance:
            self._fill_chance(node)
        else:
            self._fill_uniformly(node)

        for i in range(0, len(node.children)):
            self._fill_uniform_dfs(node.children[i])

    # --- Fills a chance node with the probability of each outcome.
    # -- @param node the chance node
    # -- @local
    @staticmethod
    def _fill_chance(node):
        assert not node.terminal

        node.strategy = arguments.Tensor(len(node.children), game_settings.hand_count).fill_(0)
        # setting probability of impossible hands to 0
        for i in range(0, len(node.children)):
            child_node = node.children[i]
            mask = card_tools.get_possible_hand_indexes(child_node.board).byte()
            node.strategy[i].fill_(0)
            # remove 4 as in Hold'em each player holds one card
            node.strategy[i][mask] = 1.0 / (game_settings.card_count - 4)

    # --- Fills a player node with a uniform strategy.
    # -- @param node the player node
    # -- @local
    @staticmethod
    def _fill_uniformly(node):
        assert  node.current_player == constants.Players.P1 or node.current_player == constants.Players.P2

        if not node.terminal:
            node.strategy = arguments.Tensor(len(node.children), game_settings.hand_count).fill_(1.0 / len(node.children))
