
import torch

import settings.arguments as arguments
import settings.game_settings as game_settings


class BetSizing(object):

    pot_fractions: list

    # --- Constructor
    # -- @param pot_fractions a list of fractions of the pot which are allowed
    # -- as bets, sorted in ascending order
    def __init__(self, pot_fractions):
        self.pot_fractions = pot_fractions or [1]

    # --- Gives the bets which are legal at a game state.
    # -- @param node a representation of the current game state, with fields:
    # --
    # -- * `bets`: the number of chips currently committed by each player
    # --
    # -- * `current_player`: the currently acting player
    # -- @return an Nx2 tensor where N is the number of new possible game states,
    # -- containing N sets of new commitment levels for each player
    def get_possible_bets(self, node):
        current_player = node.current_player.value
        assert current_player == 0 or current_player == 1, 'Wrong player for bet size computation'
        opponent = 1 - node.current_player.value
        opponent_bet = node.bets[opponent]
        assert node.bets[current_player] <= opponent_bet, "Not a betting situation"

        # compute min possible raise size
        max_raise_size = game_settings.stack - opponent_bet
        min_raise_size = opponent_bet - node.bets[current_player]
        min_raise_size = max(min_raise_size, game_settings.ante)
        min_raise_size = min(max_raise_size, min_raise_size)

        if min_raise_size == 0:
            return torch.tensor(0)   # hack to create 0-dimensional tensor
        elif min_raise_size == max_raise_size:
            out = arguments.Tensor(1, 2).fill_(opponent_bet)
            out[0][current_player] = opponent_bet + min_raise_size
            return out
        else:
            # iterate through all bets and check if they are possible
            fractions = []
            if node.num_bets == 0:
                fractions = self.pot_fractions[0]
            elif node.num_bets == 1:
                fractions = self.pot_fractions[1]
            else:
                fractions = self.pot_fractions[2]

            max_possible_bets_count = len(fractions) + 1    # we can always go allin
            out = arguments.Tensor(max_possible_bets_count, 2).fill_(opponent_bet)

            # take pot size after opponent bet is called
            pot = opponent_bet * 2
            used_bets_count = -1

            # try all pot fractions bet and see if we can use them
            for i in range(0, len(fractions)):
                raise_size = pot * fractions[i]
                if min_raise_size <= raise_size < max_raise_size:
                    used_bets_count = used_bets_count + 1
                    out[used_bets_count, current_player] = opponent_bet + raise_size

            # adding allin
            used_bets_count = used_bets_count + 1
            assert used_bets_count <= max_possible_bets_count, "Wrong number of bets"
            out[used_bets_count, current_player] = opponent_bet + max_raise_size

            return out[0:used_bets_count + 1, :]
