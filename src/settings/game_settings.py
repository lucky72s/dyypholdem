from math import comb

# definition of a deck of poker cards
suit_count = 4
rank_count = 13
card_count = suit_count * rank_count

# names for suit and rank of poker card
suit_table = ['c', 'd', 'h', 's']
rank_table = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

# hole cards per hand
hand_card_count = 2
hand_count = comb(card_count, hand_card_count)

# board structure for texas hold'em
board_card_count = [0, 3, 4, 5]

# the blind structure, in chips
ante = 100
small_blind = 50
big_blind = 100
# the size of each player's stack, in chips
stack = 20000
# list of pot-scaled bet sizes to use in tree
# orig: bet_sizing = [[1], [1], [1]]
bet_sizing = [[1], [1], [1]]

