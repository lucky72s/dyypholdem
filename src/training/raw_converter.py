import os
import sys
import argparse
sys.path.append(os.getcwd())


def convert(street: int):

    arguments.timer.start()

    bucket_conversion = BucketConversion()

    path = arguments.training_data_path

    src_folder = path + arguments.street_folders[street] + arguments.training_data_raw
    dest_folder = path + arguments.street_folders[street] + arguments.training_data_converted

    good_files = {}
    num_files = 0
    input_files = {f[0:f.find(arguments.inputs_extension)] for f in os.listdir(src_folder) if
                   f.endswith(arguments.inputs_extension)}
    target_files = {f[0:f.find(arguments.targets_extension)] for f in os.listdir(src_folder) if
                    f.endswith(arguments.targets_extension)}
    for file in input_files:
        if file in target_files:
            good_files[file] = 1
            num_files += 1

    arguments.logger.info(f"{num_files} file pairs to be converted")
    bucket_count = bucketer.get_bucket_count(street)
    target_size = bucket_count * constants.players_count
    input_size = bucket_count * constants.players_count + 1

    num_train = math.floor(num_files * 0.9)
    num_valid = num_files - num_train

    file_idx = 0
    file_pattern = r""
    if street == 2:
        file_pattern = r"[-](......)"
    elif street == 3:
        file_pattern = r"[-](........)"
    elif street == 4:
        file_pattern = r"[-](..........)"

    input_batch = arguments.Tensor(arguments.gen_batch_size, input_size)
    target_batch = arguments.Tensor(arguments.gen_batch_size, target_size)

    for file_base in good_files:
        input_name = file_base + arguments.inputs_extension
        target_name = file_base + arguments.targets_extension

        if street > 1:
            board = card_to_string.string_to_board(re.compile(file_pattern).search(file_base).groups()[0])
            bucket_conversion.set_board(board)
        else:
            bucket_conversion.set_board(arguments.Tensor())

        arguments.logger.trace(f"Loading file '{src_folder + file_base}' for conversion")
        raw_input_batch = torch.load(src_folder + input_name).type(arguments.Tensor)
        raw_target_batch = torch.load(src_folder + target_name).type(arguments.Tensor)

        raw_indexes = [[0, game_settings.hand_count], [game_settings.hand_count, game_settings.hand_count * 2]]
        bucket_indexes = [[0, bucket_count], [bucket_count, bucket_count * 2]]

        for player in range(0, constants.players_count):
            player_index = raw_indexes[player]
            bucket_index = bucket_indexes[player]
            bucket_conversion.card_range_to_bucket_range(raw_input_batch[:, player_index[0]:player_index[1]],
                                                         input_batch[:, bucket_index[0]:bucket_index[1]])

        for player in range(0, constants.players_count):
            player_index = raw_indexes[player]
            bucket_index = bucket_indexes[player]
            bucket_conversion.hand_cfvs_to_bucket_cfvs(raw_input_batch[:, player_index[0]:player_index[1]],
                                                       raw_target_batch[:, player_index[0]:player_index[1]],
                                                       input_batch[:, bucket_index[0]:bucket_index[1]],
                                                       target_batch[:, bucket_index[0]:bucket_index[1]])
        input_batch[:, -1].copy_(raw_input_batch[:, -1])

        arguments.logger.trace(f"Saving converted file '{dest_folder + file_base}'")
        torch.save(target_batch, dest_folder + target_name)
        torch.save(input_batch, dest_folder + input_name)

        file_idx = file_idx + 1
        if file_idx % 100 == 0:
            arguments.logger.debug(f"Progress: {file_idx} of {num_files} files completed")

    arguments.timer.stop(f"Conversion of {file_idx} files completed:", log_level="SUCCESS")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert raw training data into bucketed data for the specified street')
    parser.add_argument('street', type=int, choices=[1, 2, 3, 4], help="Street (1=pre-flop, 2=flop, 3=turn, 4=river)")
    args = parser.parse_args()

    import math
    import re
    import torch

    import settings.arguments as arguments
    import settings.constants as constants
    import settings.game_settings as game_settings

    import game.card_to_string_conversion as card_to_string
    from nn.bucket_conversion import BucketConversion
    import nn.bucketer as bucketer

    street_arg = int(sys.argv[1])

    arguments.logger.info(f"Converting data for street: {street_arg}")

    convert(street_arg)
