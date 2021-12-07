import math

import settings.game_settings as game_settings

base_values_pow7: []
base_values_pow6: []
base_values_pow5: []
base_values_pow4: []
base_values_pow3: []
base_values_pow2: []
base_values_pow1: []


def initialize():
    global base_values_pow7
    global base_values_pow6
    global base_values_pow5
    global base_values_pow4
    global base_values_pow3
    global base_values_pow2
    global base_values_pow1

    base_values_pow7 = [0] * game_settings.card_count
    base_values_pow6 = [0] * game_settings.card_count
    base_values_pow5 = [0] * game_settings.card_count
    base_values_pow4 = [0] * game_settings.card_count
    base_values_pow3 = [0] * game_settings.card_count
    base_values_pow2 = [0] * game_settings.card_count
    base_values_pow1 = [0] * game_settings.card_count

    for i in range(game_settings.card_count):
        base_values_pow7[i] = math.floor(i / 4) * 13 * 13 * 13 * 13 * 13 * 13
        base_values_pow6[i] = math.floor(i / 4) * 13 * 13 * 13 * 13 * 13
        base_values_pow5[i] = math.floor(i / 4) * 13 * 13 * 13 * 13
        base_values_pow4[i] = math.floor(i / 4) * 13 * 13 * 13
        base_values_pow3[i] = math.floor(i / 4) * 13 * 13
        base_values_pow2[i] = math.floor(i / 4) * 13
        base_values_pow1[i] = math.floor(i / 4)


initialize()


def suitcat_river(s1, s2, s3, s4, s5, s6, s7):
    suit = {0: 0, 1: 0, 2: 0, 3: 0}
    suit[s3] = suit[s3] + 1
    suit[s4] = suit[s4] + 1
    suit[s5] = suit[s5] + 1
    suit[s6] = suit[s6] + 1
    suit[s7] = suit[s7] + 1

    if suit[0] <= 2 and suit[1] <= 2 and suit[2] <= 2 and suit[3] <= 2:
        return 0

    if suit[0] == 3 or suit[1] == 3 or suit[2] == 3 or suit[3] == 3:
        the_suit = -1
        for i in range(0, 4):
            if suit[i] == 3:
                the_suit = i
        mask = 0
        if s3 == the_suit: mask = mask + 1
        if s4 == the_suit: mask = mask + 2
        if s5 == the_suit: mask = mask + 4
        if s6 == the_suit: mask = mask + 8
        if s7 == the_suit: mask = mask + 16

        add = 0
        if s1 == the_suit and s2 == the_suit:
            add = 1
        elif s1 == the_suit:
            add = 2
        elif s2 == the_suit:
            add = 3

        if mask == 7:
            return 1 + add
        elif mask == 11:
            return 5 + add
        elif mask == 19:
            return 9 + add
        elif mask == 13:
            return 13 + add
        elif mask == 21:
            return 17 + add
        elif mask == 25:
            return 21 + add
        elif mask == 14:
            return 25 + add
        elif mask == 22:
            return 29 + add
        elif mask == 26:
            return 33 + add
        elif mask == 28:
            return 37 + add

        raise ValueError("bad river suits")

    if suit[0] == 4 or suit[1] == 4 or suit[2] == 4 or suit[3] == 4:
        the_suit = -1
        for i in range(0, 4):
            if suit[i] == 4:
                the_suit = i
        if s3 != the_suit:
            if s1 == the_suit and s2 == the_suit: return 42
            if s1 == the_suit: return 43
            if s2 == the_suit: return 44
            return 45
        elif s4 != the_suit:
            if s1 == the_suit and s2 == the_suit: return 46
            if s1 == the_suit: return 47
            if s2 == the_suit: return 48
            return 49
        elif s5 != the_suit:
            if s1 == the_suit and s2 == the_suit: return 50
            if s1 == the_suit: return 51
            if s2 == the_suit: return 52
            return 53
        elif s6 != the_suit:
            if s1 == the_suit and s2 == the_suit: return 54
            if s1 == the_suit: return 55
            if s2 == the_suit: return 56
            return 57
        elif s7 != the_suit:
            if s1 == the_suit and s2 == the_suit: return 58
            if s1 == the_suit: return 59
            if s2 == the_suit: return 60
            return 61

        raise ValueError("bad river suits")

    if suit[0] == 5 or suit[1] == 5 or suit[2] == 5 or suit[3] == 5:
        the_suit = -1
        for i in range(0, 3):
            if suit[i] == 5:
                the_suit = i
        if s1 == the_suit and s2 == the_suit: return 62
        if s1 == the_suit: return 63
        if s2 == the_suit: return 64
        return 65

    raise ValueError("bad river suits")


def river_id(hand):
    base_value = base(hand)
    suit_code = suit(hand)
    if suit_code == -1:
        raise ValueError("invalid suit code")
    suit_code = suit_code * 815730722

    return base_value + suit_code


def base(hand):
    v1 = base_values_pow7[hand[0]]
    v2 = base_values_pow6[hand[1]]
    v3 = base_values_pow5[hand[2]]
    v4 = base_values_pow4[hand[3]]
    v5 = base_values_pow3[hand[4]]
    v6 = base_values_pow2[hand[5]]
    v7 = base_values_pow1[hand[6]]
    return v1 + v2 + v3 + v4 + v5 + v6 + v7


def suit(hand):
    return suitcat_river((hand[0] + 1) % 4,
                         (hand[1] + 1) % 4,
                         (hand[2] + 1) % 4,
                         (hand[3] + 1) % 4,
                         (hand[4] + 1) % 4,
                         (hand[5] + 1) % 4,
                         (hand[6] + 1) % 4)

