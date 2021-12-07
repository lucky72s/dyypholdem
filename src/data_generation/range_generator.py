
import torch

import settings.arguments as arguments
import settings.game_settings as game_settings
import game.card_tools as card_tools
from game.evaluation.evaluator import Evaluator
from terminal_equity.terminal_equity import TerminalEquity

import utils.pseudo_random as random_


class RangeGenerator(object):

    possible_hands_count: int
    possible_hands_mask: torch.Tensor
    reverse_order: torch.Tensor
    reordered_range: torch.Tensor
    sorted_range: torch.Tensor

    def __init__(self):
        return

    # -- Recursively samples a section of the range vector.
    # -- @param cards an NxJ section of the range tensor, where N is the batch size
    # -- and J is the length of the range sub-vector
    # -- @param mass a vector of remaining probability mass for each batch member
    # -- @see generate_range
    # -- @local
    def _generate_recursion(self, cards_range, mass):
        batch_size = cards_range.shape[0]
        assert mass.shape[0] == batch_size

        # recursion stops at 1
        card_count = cards_range.shape[1]
        if card_count == 1:
            cards_range.copy_(mass.view_as(cards_range))
        else:
            rand = random_.rand(batch_size)
            mass1 = torch.clone(mass).mul_(rand)
            mass1[torch.lt(mass1, 0.00001)] = 0
            mass1[torch.gt(mass1, 0.99999)] = 1
            mass2 = mass - mass1
            half_size = card_count / 2
            if half_size % 1 != 0:
                half_size = half_size - 0.5
                half_size = half_size + random_.randint(0, 1)
            self._generate_recursion(cards_range[:, 0:int(half_size)], mass1)
            self._generate_recursion(cards_range[:, int(half_size):], mass2)

    # --- Samples a batch of ranges with hands sorted by strength on the board.
    # -- @param range a NxK tensor in which to store the sampled ranges, where N is
    # -- the number of ranges to sample and K is the range size
    # -- @see generate_range
    # -- @local
    def _generate_sorted_range(self, cards_range):
        batch_size = cards_range.size(0)
        self._generate_recursion(cards_range, arguments.Tensor(batch_size).fill_(1))

    # --- Samples a batch of random range vectors.
    # --
    # -- Each vector is sampled indepently by randomly splitting the probability
    # -- mass between the bottom half and the top half of the range, and then
    # -- recursing on the two halfs.
    # --
    # -- @{set_board} must be called first.
    # --
    # -- @param range a NxK tensor in which to store the sampled ranges, where N is
    # -- the number of ranges to sample and K is the range size
    def generate_range(self, cards_range: torch.Tensor):
        batch_size = cards_range.size(0)
        self.sorted_range.resize_([batch_size, self.possible_hands_count])
        self._generate_sorted_range(self.sorted_range)

        # we have to reorder the the range back to undo the sort by strength
        index = self.reverse_order.expand_as(self.sorted_range)
        self.reordered_range = self.sorted_range.gather(1, index)
        cards_range.zero_()
        cards_range.masked_scatter_(self.possible_hands_mask.expand_as(cards_range), self.reordered_range)

    # --- Sets the (possibly empty) board cards to sample ranges with.
    # --
    # -- The sampled ranges will assign 0 probability to any private hands that
    # -- share any cards with the board.
    # --
    # -- @param board a possibly empty vector of board cards
    def set_board(self, te: TerminalEquity, board):
        hand_strengths = arguments.Tensor(game_settings.hand_count)
        for i in range(0, game_settings.hand_count):
            hand_strengths[i] = i
        if board.dim() == 0:
            raise NotImplementedError()
        elif board.size(0) == 5:
            hand_strengths = Evaluator.batch_eval(board, None)
        else:
            hand_strengths = te.get_hand_strengths().squeeze()

        possible_hand_indexes = card_tools.get_possible_hand_indexes(board)
        self.possible_hands_count = int(possible_hand_indexes.sum(0).item())
        possible_hands_mask = possible_hand_indexes.view(1, -1)
        self.possible_hands_mask = possible_hands_mask.bool()

        non_colliding_strengths = hand_strengths.masked_select(self.possible_hands_mask)
        order = non_colliding_strengths.sort()

        # hack to create same sort order even with duplicate values
        if arguments.use_gpu:
            non_colliding_strengths2 = torch.cuda.DoubleTensor(self.possible_hands_count).zero_()
        else:
            non_colliding_strengths2 = torch.DoubleTensor(self.possible_hands_count).zero_()
        for i in range(0, self.possible_hands_count):
            non_colliding_strengths2[i] = float(non_colliding_strengths[i]) - (i / 10000)
        order2 = non_colliding_strengths2.sort()
        reverse_order = order2.indices.clone().sort().indices.clone()
        self.reverse_order = reverse_order.view(1, -1).to(torch.long)

        self.reordered_range = arguments.Tensor()
        self.sorted_range = arguments.Tensor()
