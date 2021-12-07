
import torch

import settings.arguments as arguments
import settings.game_settings as game_settings

import utils.pseudo_random as random_


# -- Samples a random set of cards.
# --
# -- Each subset of the deck of the correct size is sampled with
# -- uniform probability.
# --
# -- @param count the number of cards to sample
# -- @return a vector of cards, represented numerical
def generate_cards(count):
    # marking all used cards
    used_cards = torch.ByteTensor(game_settings.card_count).zero_()

    out = arguments.Tensor(count)
    generated_cards_count = 0
    while generated_cards_count < count:
        card = random_.randint(0, game_settings.card_count - 1)
        if used_cards[card] == 0:
            out[generated_cards_count] = card
            generated_cards_count += 1
            used_cards[card] = 1
    return out.int()

