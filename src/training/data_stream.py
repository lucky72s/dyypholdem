import os
import math

import torch

import settings.arguments as arguments
import settings.constants as constants

import nn.bucketer as bucketer

import utils.pseudo_random as random_


class DataStream(object):

    def __init__(self, street):

        path = arguments.training_data_path

        self.src_folder = path + arguments.street_folders[street] + arguments.training_data_converted

        self.good_files = []
        num_files = 0
        input_files = {f[0:f.find(arguments.inputs_extension)] for f in os.listdir(self.src_folder) if
                       f.endswith(arguments.inputs_extension)}
        target_files = {f[0:f.find(arguments.targets_extension)] for f in os.listdir(self.src_folder) if
                        f.endswith(arguments.targets_extension)}
        for file in input_files:
            if file in target_files:
                self.good_files.append(file)
                num_files += 1
        arguments.logger.debug(f"{num_files} good files found for training")

        self.bucket_count = bucketer.get_bucket_count(street)
        self.target_size = self.bucket_count * constants.players_count
        self.input_size = self.bucket_count * constants.players_count + 1

        num_train = math.floor(num_files * 0.9)
        num_valid = num_files - num_train

        train_count = num_train * arguments.gen_batch_size
        valid_count = num_valid * arguments.gen_batch_size

        self.train_data_count = train_count
        assert self.train_data_count >= arguments.train_batch_size, 'Training data count has to be greater than a train batch size!'
        self.train_batch_count = int(self.train_data_count / arguments.train_batch_size)
        self.valid_data_count = valid_count
        assert self.valid_data_count >= arguments.train_batch_size, 'Validation data count has to be greater than a train batch size!'
        self.valid_batch_count = int(self.valid_data_count / arguments.train_batch_size)

    # --- Randomizes the order of training data.
    # --
    # -- Done so that the data is encountered in a different order for each epoch.
    def start_epoch(self):
        # data are shuffled each epoch]
        self.shuffle(self.good_files, int(self.train_data_count / arguments.gen_batch_size))

    @staticmethod
    def shuffle(tbl: list, n):
        if arguments.use_pseudo_random:
            tbl.sort()
            return tbl
        else:
            for i in range(n, 0, -1):
                rand = random_.randint(1, n)
                tbl[i], tbl[rand] = tbl[rand], tbl[i]
        return tbl

    # --- Gives the number of batches of training data.
    # --
    # -- Batch size is defined by @{arguments.train_batch_size}
    # -- @return the number of batches
    def get_training_batch_count(self):
        return self.train_batch_count

    # --- Gives the number of batches of validation data.
    # --
    # -- Batch size is defined by @{arguments.train_batch_size}.
    # -- @return the number of batches
    def get_validation_batch_count(self):
        return self.valid_batch_count

    # --- Returns a batch of data from the training set.
    # -- @param batch_index the index of the batch to return
    # -- @return the inputs set for the batch
    # -- @return the targets set for the batch
    # -- @return the masks set for the batch
    def get_training_batch(self, batch_index):
        return self.get_batch(batch_index)

    # --- Returns a batch of data from the validation set.
    # -- @param batch_index the index of the batch to return
    # -- @return the inputs set for the batch
    # -- @return the targets set for the batch
    # -- @return the masks set for the batch
    def get_validation_batch(self, batch_index):
        return self.get_batch(self.train_batch_count + batch_index)

    # --- Returns a batch of data from a specified data set.
    # -- @param inputs the inputs set for the given data set
    # -- @param targets the targets set for the given data set
    # -- @param mask the masks set for the given data set
    # -- @param batch_index the index of the batch to return
    # -- @return the inputs set for the batch
    # -- @return the targets set for the batch
    # -- @return the masks set for the batch
    # -- @local
    def get_batch(self, batch_index) -> {torch.Tensor, torch.Tensor, torch.Tensor}:

        inputs = arguments.Tensor(arguments.train_batch_size, self.input_size)
        targets = arguments.Tensor(arguments.train_batch_size, self.target_size)
        masks = arguments.Tensor(arguments.train_batch_size, self.target_size).zero_()

        for i in range(int(arguments.train_batch_size / arguments.gen_batch_size)):

            idx = batch_index * arguments.train_batch_size / arguments.gen_batch_size + i
            idx = math.floor(idx + 0.1)
            file_base = self.good_files[idx]

            input_name = file_base + ".inputs"
            target_name = file_base + ".targets"

            input_batch = torch.load(self.src_folder + input_name)
            target_batch = torch.load(self.src_folder + target_name)

            data_index = [i * arguments.gen_batch_size, (i + 1) * arguments.gen_batch_size]

            inputs[data_index[0]:data_index[1], :].copy_(input_batch)
            targets[data_index[0]:data_index[1], :].copy_(target_batch)
            masks[data_index[0]:data_index[1]][torch.gt(input_batch[:, 0:self.bucket_count * 2], 0)] = 1

        return inputs, targets, masks
