import gc
import time
import math

import torch

import settings.arguments as arguments
import settings.game_settings as game_settings
import settings.constants as constants
from terminal_equity.terminal_equity import TerminalEquity
import random_card_generator as card_generator
import game.card_to_string_conversion as card_to_string_conversion
from lookahead.resolving import Resolving
from tree.tree_node import TreeNode

import utils.pseudo_random as random_

arguments.timer.start("Preparing data generation environment", log_level="DEBUG")
from range_generator import RangeGenerator
arguments.timer.stop("Preparation completed", log_level="DEBUG")


class DataGenerator:

    def __init__(self):
        pass

    # -- Generates training files by sampling random poker situations and solving them.
    # --
    # -- @param train_data_count the number of training examples to generate
    def generate_data(self, gen_data_count, street) -> None:

        if arguments.use_pseudo_random:
            random_.manual_seed(123)

        arguments.timer.start("Starting data generation", log_level="DEBUG")

        self.generate_data_file(gen_data_count, street)

        arguments.timer.stop("Data generation completed in:", log_level="SUCCESS")

    # -- Generates data files containing examples of random poker situations with
    # -- counterfactual values from an associated solution.
    # --
    # -- Each poker situation is randomly generated using @{range_generator} and
    # -- @{random_card_generator}. For description of neural net input and target
    # -- type, see @{net_builder}.
    # --
    # -- @param data_count the number of examples to generate
    # -- @param file_name the prefix of the files where the data is saved (appended with `.inputs`, `.targets`, and `.mask`).
    @staticmethod
    def generate_data_file(data_count, street):
        range_generator = RangeGenerator()
        batch_size = arguments.gen_batch_size
        assert data_count % batch_size == 0, 'data count has to be divisible by the batch size'
        batch_count = data_count / batch_size

        target_size = game_settings.hand_count * constants.players_count
        targets = arguments.Tensor(batch_size, target_size)
        input_size = game_settings.hand_count * constants.players_count + 1
        inputs = arguments.Tensor(batch_size, input_size)
        mask = arguments.Tensor(batch_size, game_settings.hand_count).zero_()

        terminal_equity = TerminalEquity()

        for batch in range(0, int(batch_count)):

            arguments.timer.split_start(f"Starting batch {batch+1}", log_level="DEBUG")

            board = card_generator.generate_cards(game_settings.board_card_count[street - 1])
            board_string = card_to_string_conversion.cards_to_string(board)

            terminal_equity.set_board(board)
            range_generator.set_board(terminal_equity, board)

            # generating ranges
            ranges = arguments.Tensor(constants.players_count, batch_size, game_settings.hand_count).zero_()
            for player in range(0, constants.players_count):
                range_generator.generate_range(ranges[player])

            min_pot = [100, 200, 400, 2000, 6000]
            max_pot = [100, 400, 2000, 6000, 18000]
            pot_range = []
            for i in range(0, len(min_pot)):
                pot_range.append(max_pot[i] - min_pot[i])

            random_pot_cat = int(random_.rand(1).mul_(len(min_pot)).add_(0).floor()[0].item())
            random_pot_size = random_.rand(1)[0].item()
            random_pot_size = random_pot_size * pot_range[random_pot_cat]
            random_pot_size = random_pot_size + min_pot[random_pot_cat]
            random_pot_size = math.floor(random_pot_size)

            pot_size_feature = (random_pot_size / game_settings.stack)

            # translating ranges to features
            pot_feature_index = -1
            inputs[:, pot_feature_index].fill_(pot_size_feature)

            player_indexes = [[0, game_settings.hand_count], [game_settings.hand_count, game_settings.hand_count * 2]]
            for player in range(0, constants.players_count):
                player_index = player_indexes[player]
                inputs[:, player_index[0]:player_index[1]].copy_(ranges[player])

            # computation of values using re-solving
            values = arguments.Tensor(batch_size, constants.players_count, game_settings.hand_count)
            pot_size = random_pot_size
            arguments.logger.debug(f"Generating data for Board: {board_string} and Pot: {pot_size}")

            arguments.timer.split_start("Start resolving", log_level="TRACE")
            resolving = Resolving(terminal_equity)

            current_node = TreeNode()
            current_node.street = street
            current_node.board = board
            current_node.board_string = board_string
            current_node.current_player = street == 1 and constants.Players.P1 or constants.Players.P2
            current_node.bets = arguments.Tensor([pot_size, pot_size])

            p1_range = ranges[0]
            p2_range = ranges[1]
            resolving.resolve_first_node(current_node, p1_range, p2_range)

            root_values = resolving.get_root_cfv_both_players()
            root_values.mul_(1 / pot_size)
            values.copy_(root_values)
            for player in range(0, constants.players_count):
                player_index = player_indexes[player]
                targets[:, player_index[0]:player_index[1]].copy_(values[:, player, :])

            arguments.logger.trace(f"Resolving completed - target sum: {targets.sum():.6f}")

            file_time = str(int(time.time()))
            basename = f"{file_time}-{board_string}"
            file_path = arguments.training_data_path + arguments.street_folders[street] + arguments.training_data_raw + basename

            arguments.logger.trace(f"Saving generated data file '{file_path}'")
            torch.save(inputs, file_path + arguments.inputs_extension)
            torch.save(targets, file_path + arguments.targets_extension)

            arguments.timer.split_stop("Batch completed in", log_level="TIMING")

            # forced clean-up
            del resolving
            del current_node
            if arguments.use_gpu:
                arguments.logger.trace(f"Initiating garbage collection. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")
            gc.collect()
            if arguments.use_gpu:
                torch.cuda.empty_cache()
                arguments.logger.trace(f"Garbage collection completed. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")
