from math import floor

import torch

import settings.arguments as arguments
import settings.game_settings as game_settings


# -- Gets the suit of a card.
# -- @param card the numeric representation of the card
# -- @return the index of the suit
def card_to_suit(card: int):
    return card % game_settings.suit_count


# -- Gets the rank of a card.
# -- @param card the numeric representation of the card
# -- @return the index of the rank
def card_to_rank(card):
    return floor(card / game_settings.suit_count)


card_to_string_table = [""] * game_settings.card_count
for a_card in range(0, game_settings.card_count):
    rank_name = game_settings.rank_table[card_to_rank(a_card)]
    suit_name = game_settings.suit_table[card_to_suit(a_card)]
    card_to_string_table[a_card] = rank_name + suit_name

string_to_card_table = {}
for a_card in range(0, game_settings.card_count):
    string_to_card_table[card_to_string_table[a_card]] = a_card


# -- Converts a card's numeric representation to its string representation.
# -- @param card the numeric representation of a card
# -- @return the string representation of the card
def card_to_string(card):
    assert (0 <= card < game_settings.card_count)
    return card_to_string_table[card]


# -- Converts several cards' numeric representations to their string
# -- representations.
# -- @param cards a vector of numeric representations of cards
# -- @return a string containing each card's string representation, concatenated
def cards_to_string(cards: torch.Tensor) -> str:
    if cards.dim() == 0:
        return ""

    out = ""
    for card in range(0, cards.size(0)):
        out = out + card_to_string(cards[card].int())
    return out


# --- Converts a string representing zero or more board cards to a
# -- vector of numeric representations.
# -- @param card_string either the empty string or a string representation of a
# -- card
# -- @return either an empty tensor or a tensor containing the numeric
# -- representation of the card
def string_to_board(card_string):
    # assert card_string
    if card_string == '':
        return arguments.Tensor()

    num_cards = int(len(card_string) / 2)
    board = arguments.Tensor(num_cards)
    for i in range(0, num_cards):
        board[i] = string_to_card(card_string[i * 2: (i + 1) * 2])
    return board.long()


# --- Converts a card's string representation to its numeric representation.
# -- @param card_string the string representation of a card
# -- @return the numeric representation of the card
def string_to_card(card_string: str):
    card_index = string_to_card_table[card_string]
    assert 0 <= card_index < game_settings.card_count
    return card_index

