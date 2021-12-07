import math
from typing import List

import settings.game_settings as game_settings


base_values_pow5: []
base_values_pow4: []
base_values_pow3: []
base_values_pow2: []
base_values_pow1: []
scale_factor = 13 ** 5


def initialize():
    global base_values_pow5
    global base_values_pow4
    global base_values_pow3
    global base_values_pow2
    global base_values_pow1

    base_values_pow5 = [0] * game_settings.card_count
    base_values_pow4 = [0] * game_settings.card_count
    base_values_pow3 = [0] * game_settings.card_count
    base_values_pow2 = [0] * game_settings.card_count
    base_values_pow1 = [0] * game_settings.card_count

    for i in range(game_settings.card_count):
        base_values_pow5[i] = math.floor(i / 4) * 13 * 13 * 13 * 13
        base_values_pow4[i] = math.floor(i / 4) * 13 * 13 * 13
        base_values_pow3[i] = math.floor(i / 4) * 13 * 13
        base_values_pow2[i] = math.floor(i / 4) * 13
        base_values_pow1[i] = math.floor(i / 4)


initialize()


def _suitcat_turn(s1, s2, s3, s4, s5):

    if s1 != 0:
        return -1

    ret = -1
    if s2 == 0:
        if s3 == 0:
            ret = s4 * 2 + s5
        elif s3 == 1:
            ret = 5 + s4 * 3 + s5
    elif s2 == 1:
        if s3 == 0:
            ret = 15 + s4 * 3 + s5
        elif s3 == 1:
            ret = 25 + s4 * 3 + s5
        elif s3 == 2:
            ret = 35 + s4 * 4 + s5
    return ret


def flop_id(hand: List):
    # Get hand suits
    os = [0] * 5
    for i in range(5):
        os[i] = hand[i] % 4

    # Canonicalize suits
    MM = 0
    s = [0] * 5
    for i in range(5):
        j = 0
        while j < i:
            if os[i] == os[j]:
                s[i] = s[j]
                break
            j += 1
        if j == i:
            s[i] = MM
            MM += 1
        hand[i] += s[i] - (hand[i] % 4)
    hole = hand[0:2]
    board = hand[2:6]
    board.sort()
    hand = hole + board

    base_value = base(hand)

    for i in range(5):
        s[i] = hand[i] % 4

    cat = _suitcat_turn(s[0], s[1], s[2], s[3], s[4])
    assert cat != -1, "wrong flop cat"
    cat = cat * scale_factor + base_value

    return cat


def base(hand):
    v1 = base_values_pow5[hand[0]]
    v2 = base_values_pow4[hand[1]]
    v3 = base_values_pow3[hand[2]]
    v4 = base_values_pow2[hand[3]]
    v5 = base_values_pow1[hand[4]]
    return v1 + v2 + v3 + v4 + v5
