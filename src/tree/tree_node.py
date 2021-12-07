from dataclasses import dataclass

import torch

import settings.arguments as arguments
import settings.constants as constants

from game.bet_sizing import BetSizing


@dataclass
class TreeNode:
    depth: int = 0
    street: int = 0
    board: arguments.Tensor = None
    board_string: str = ""
    current_player: constants.Players = 0
    bets: arguments.Tensor = None
    num_bets: int = 0
    terminal: bool = False
    type: constants.NodeTypes = constants.NodeTypes.undefined
    parent: object = None
    children: [] = None
    actions: arguments.Tensor = None
    strategy: arguments.Tensor = None
    bet_sizing: BetSizing = None
    pot: torch.Tensor = None
    lookahead_coordinates: arguments.Tensor = None

    def __repr__(self, level=0):
        if level > 3:
            return ''
        if level == 0:
            header = "Decision Tree:\n"
            indent = "  |> "
        else:
            header = ''
            indent = "  " + "    " * (level - 1) + "|---> "
        ret = f"{header}{indent}Type={self.type}, depth={self.depth}, street={arguments.street_names[self.street]}, player={repr(self.current_player)}, bets=({self.bets[0].item()}, {self.bets[1].item()}), pot={self.pot}\n"
        if self.children:
            for child in self.children:
                ret += child.__repr__(level + 1)
        return ret


@dataclass
class BuildTreeParams:
    root_node: TreeNode
    limit_to_street: bool
    bet_sizing: BetSizing = None

