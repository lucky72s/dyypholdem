
import settings.arguments as arguments


class ResolveResults(object):

    strategy: arguments.Tensor
    achieved_cfvs: arguments.Tensor
    root_cfvs: arguments.Tensor
    root_cfvs_both_players: arguments.Tensor
    children_cfvs: arguments.Tensor
    actions: list

    def __init__(self):
        pass

    def get_cfv(self, player: int, pocket_index: int) -> float:
        return self.root_cfvs_both_players[player, pocket_index].item()

    def get_actions(self):
        return self.actions

    def get_player_strategy(self, action_index: int, pocket_index: int) -> float:
        return self.strategy[action_index, 0, pocket_index].item()
