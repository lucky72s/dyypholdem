import time

import torch

import settings.arguments as arguments
import settings.game_settings as game_settings
import settings.constants as constants

from nn.bucket_conversion import BucketConversion
from nn.next_round_value_pre import NextRoundValuePre
from nn.value_nn import ValueNn
import nn.bucketer as bucketer
from terminal_equity.terminal_equity import TerminalEquity

import utils.pseudo_random as random_

arguments.timer.start("Preparing data generation environment", log_level="DEBUG")
from range_generator import RangeGenerator
arguments.timer.stop("Preparation completed", log_level="DEBUG")


class DataGeneratorAux:

    def __init__(self):
        pass

    # -- Generates training files by sampling random poker situations and solving them.
    # --
    # -- @param train_data_count the number of training examples to generate
    def generate_data(self, gen_data_count, street):

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

        board = arguments.Tensor()
        terminal_equity = TerminalEquity()
        terminal_equity.set_board(board)
        range_generator.set_board(terminal_equity, board)

        bucket_conversion = BucketConversion()
        bucket_conversion.set_board(board)

        next_round = NextRoundValuePre(ValueNn().load_for_street(street), None, board)

        bucket_count = bucketer.get_bucket_count(street)
        bucketed_target_size = bucket_count * constants.players_count
        bucketed_input_size = bucket_count * constants.players_count + 1

        input_batch = arguments.Tensor(arguments.gen_batch_size, bucketed_input_size)
        target_batch = arguments.Tensor(arguments.gen_batch_size, bucketed_target_size)

        raw_indexes = [[0, game_settings.hand_count], [game_settings.hand_count, game_settings.hand_count * 2]]
        bucket_indexes = [[0, bucket_count], [bucket_count, bucket_count * 2]]

        for batch in range(0, int(batch_count)):

            arguments.timer.split_start(f"Starting batch {batch+1}", log_level="DEBUG")

            # generating ranges
            ranges = arguments.Tensor(constants.players_count, batch_size, game_settings.hand_count).zero_()
            for player in range(0, constants.players_count):
                range_generator.generate_range(ranges[player])

            min_pot = [100, 200, 400, 2000, 6000]
            max_pot = [100, 400, 2000, 6000, 18000]
            pot_range = []
            for i in range(0, len(min_pot)):
                pot_range.append(max_pot[i] - min_pot[i])

            random_pot_cats = random_.rand(arguments.gen_batch_size)
            random_pot_cats.mul_(len(min_pot)).add_(0).floor_()
            random_pot_sizes = random_.rand(arguments.gen_batch_size)
            for i in range(arguments.gen_batch_size):
                random_pot_sizes[i] = random_pot_sizes[i] * pot_range[int(random_pot_cats[i].item())] + min_pot[int(random_pot_cats[i].item())]

            pot_size_features = random_pot_sizes.clone().mul_(1 / game_settings.stack)

            # translating ranges to features
            pot_feature_index = -1
            inputs[:, pot_feature_index].copy_(pot_size_features)
            input_batch[:, pot_feature_index].copy_(pot_size_features)

            player_indexes = [[0, game_settings.hand_count], [game_settings.hand_count, game_settings.hand_count * 2]]
            for player in range(0, constants.players_count):
                player_index = player_indexes[player]
                inputs[:, player_index[0]:player_index[1]].copy_(ranges[player])

            for i in range(arguments.gen_batch_size):
                next_street_boxes_inputs = arguments.Tensor(1, constants.players_count, game_settings.hand_count).zero_()
                next_street_boxes_outputs = next_street_boxes_inputs.clone()

                for player in range(constants.players_count):
                    player_index = player_indexes[player]
                    next_street_boxes_inputs[:, player, :].copy_(inputs[i, player_index[0]:player_index[1]])

                next_round.start_computation(random_pot_sizes[i], 1)
                next_round.get_value(next_street_boxes_inputs, next_street_boxes_outputs)

                for player in range(constants.players_count):
                    player_index = player_indexes[player]
                    targets[i, player_index[0]:player_index[1]].copy_(next_street_boxes_outputs[:, player, :].squeeze())

            for player in range(constants.players_count):
                player_index = raw_indexes[player]
                bucket_index = bucket_indexes[player]
                bucket_conversion.card_range_to_bucket_range(inputs[:, player_index[0]:player_index[1]], input_batch[:, bucket_index[0]:bucket_index[1]])

            for player in range(constants.players_count):
                player_index = raw_indexes[player]
                bucket_index = bucket_indexes[player]
                bucket_conversion.hand_cfvs_to_bucket_cfvs(inputs[:, player_index[0]:player_index[1]], targets[:, player_index[0]:player_index[1]], input_batch[:, bucket_index[0]:bucket_index[1]], target_batch[:, bucket_index[0]:bucket_index[1]])

            file_time = str(int(time.time()))
            basename = f"{file_time}-{batch+1}"
            file_path = arguments.training_data_path + arguments.street_folders[street] + arguments.training_data_raw + basename

            arguments.logger.trace(f"Saving file '{file_path}'")
            torch.save(input_batch, file_path + arguments.inputs_extension)
            torch.save(target_batch, file_path + arguments.targets_extension)

            arguments.timer.split_stop("Batch completed in", log_level="TIMING")