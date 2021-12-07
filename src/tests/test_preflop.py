
import settings.arguments as arguments
import settings.constants as constants
import settings.game_settings as game_settings

import game.card_tools as card_tools
import game.card_to_string_conversion as card_to_string
from tree.tree_node import TreeNode


def prepare_test():

    current_node = TreeNode()

    current_node.board = card_to_string.string_to_board('')
    current_node.street = 1
    current_node.current_player = constants.Players.P1
    current_node.bets = arguments.Tensor([50, 100])
    current_node.num_bets = 1

    player_range_tensor = arguments.Tensor(1, game_settings.hand_count)
    opponent_range_tensor = arguments.Tensor(1, game_settings.hand_count)

    # uniform range
    player_range_tensor[0].copy_(card_tools.get_uniform_range(current_node.board))
    opponent_range_tensor[0].copy_(card_tools.get_uniform_range(current_node.board))

    return current_node, player_range_tensor, opponent_range_tensor

