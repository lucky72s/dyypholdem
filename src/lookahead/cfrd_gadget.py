
import torch

import settings.arguments as arguments
import settings.game_settings as game_settings
import settings.constants as constants

import game.card_tools as card_tools


class CFRDGadget(object):

    # # --- Constructor
    # # -- @param board board card
    # # -- @param player_range an initial range vector for the opponent
    # # -- @param opponent_cfvs the opponent counterfactual values vector used for re-solving
    def __init__(self, board, player_range, opponent_cfvs):
        assert board is not None

        self.input_opponent_range = player_range.clone()
        self.input_opponent_value = opponent_cfvs.clone()

        self.current_opponent_values = arguments.Tensor(game_settings.hand_count)

        self.regret_epsilon = 1.0 / 100000000

        self.play_current_strategy = arguments.Tensor(game_settings.hand_count).fill_(0)
        self.terminate_current_strategy = arguments.Tensor(game_settings.hand_count).fill_(1)

        # holds achieved CFVs at each iteration so that we can compute regret
        self.total_values = arguments.Tensor(game_settings.hand_count)
        self.total_values_p2 = None

        self.terminate_regrets = arguments.Tensor(game_settings.hand_count).fill_(0)
        self.play_regrets = arguments.Tensor(game_settings.hand_count).fill_(0)

        self.regret_sum = None
        self.play_current_regret = None
        self.terminate_current_regret = None
        self.play_positive_regrets = None
        self.terminate_positive_regrets = None

        # init range mask for masking out impossible hands
        self.range_mask = card_tools.get_possible_hand_indexes(board)

    # --- Uses one iteration of the gadget game to generate an opponent range for
    # -- the current re-solving iteration.
    # -- @param current_opponent_cfvs the vector of cfvs that the opponent receives
    # -- with the current strategy in the re-solve game
    # -- @param iteration the current iteration number of re-solving
    # -- @return the opponent range vector for this iteration
    def compute_opponent_range(self, current_opponent_cfvs):

        play_values = current_opponent_cfvs
        terminate_values = self.input_opponent_value

        # 1.0 compute current regrets
        torch.mul(play_values.view(self.play_current_strategy.shape), self.play_current_strategy, out=self.total_values)
        self.total_values_p2 = self.total_values_p2 if self.total_values_p2 is not None else self.total_values.clone().zero_()
        torch.mul(terminate_values.view(self.terminate_current_strategy.shape), self.terminate_current_strategy, out=self.total_values_p2)
        self.total_values.add_(self.total_values_p2)

        self.play_current_regret = self.play_current_regret if self.play_current_regret is not None else play_values.view(self.play_current_strategy.shape).clone().zero_()
        self.terminate_current_regret = self.terminate_current_regret if self.terminate_current_regret is not None else self.play_current_regret.clone().zero_()

        self.play_current_regret.copy_(play_values.view(self.play_current_regret.shape))
        self.play_current_regret.sub_(self.total_values)

        self.terminate_current_regret.copy_(terminate_values.view(self.terminate_current_regret.shape))
        self.terminate_current_regret.sub_(self.total_values)

        # 1.1 cumulate regrets
        self.play_regrets.add_(self.play_current_regret)
        self.terminate_regrets.add_(self.terminate_current_regret)

        # 2.0 we use cfr+ in reconstruction
        self.terminate_regrets.clamp_(self.regret_epsilon, constants.max_number())
        self.play_regrets.clamp_(self.regret_epsilon, constants.max_number())

        self.play_positive_regrets = self.play_regrets
        self.terminate_positive_regrets = self.terminate_regrets

        # 3.0 regret matching
        self.regret_sum = self.regret_sum if self.regret_sum is not None else self.play_positive_regrets.clone().zero_()
        self.regret_sum.copy_(self.play_positive_regrets)
        self.regret_sum.add_(self.terminate_positive_regrets)

        self.play_current_strategy.copy_(self.play_positive_regrets)
        self.terminate_current_strategy.copy_(self.terminate_positive_regrets)

        self.play_current_strategy.div_(self.regret_sum)
        self.terminate_current_strategy.div_(self.regret_sum)

        # 4.0 for poker, the range size is larger than the allowed hands
        # we need to make sure reconstruction does not choose a range that is not allowed
        self.play_current_strategy.mul_(self.range_mask)
        self.terminate_current_strategy.mul_(self.range_mask)

        self.input_opponent_range = self.input_opponent_range if self.input_opponent_value is not None else self.play_current_strategy.clone().zero_()
        self.input_opponent_range.copy_(self.play_current_strategy)

        return self.input_opponent_range
