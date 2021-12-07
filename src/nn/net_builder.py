
import settings.arguments as arguments

import nn.bucketer as bucketer
from nn.modules.module import Module
from nn.modules.batch_norm import BatchNormalization
from nn.modules.linear import Linear
from nn.modules.prelu import PReLU
from nn.modules.sequential import Sequential
from nn.modules.narrow import Narrow
from nn.modules.concat_table import ConcatTable
from nn.modules.add_table import CAddTable
from nn.modules.dot_product import DotProduct
from nn.modules.replicate import Replicate
from nn.modules.mul_constant import MulConstant
from nn.modules.select_table import SelectTable

import utils.pseudo_random as random_


class TrainingNetwork(object):

    def __init__(self):
        pass

    @staticmethod
    def build_net(street, raw=None) -> Module:

        arguments.logger.trace("Building neural network")

        bucket_count = None
        if raw is not None:
            bucket_count = bucketer.get_rank_count()
        else:
            bucket_count = bucketer.get_bucket_count(street)

        player_count = 2
        output_size = bucket_count * player_count
        input_size = output_size + 1

        # for reproducibility
        if arguments.use_pseudo_random:
            random_.manual_seed(123)

        forward_part = Sequential()
        forward_part.add(Linear(input_size, 500))
        forward_part.add(BatchNormalization(500))
        forward_part.add(PReLU())
        forward_part.add(Linear(500, 500))
        forward_part.add(BatchNormalization(500))
        forward_part.add(PReLU())
        forward_part.add(Linear(500, 500))
        forward_part.add(BatchNormalization(500))
        forward_part.add(PReLU())
        forward_part.add(Linear(500, output_size))

        right_part = Sequential()
        right_part.add(Narrow(1, 0, output_size))

        first_layer = ConcatTable()
        first_layer.add(forward_part)
        first_layer.add(right_part)

        left_part_2 = Sequential()
        left_part_2.add(SelectTable(0))

        right_part_2 = Sequential()
        right_part_2.add(DotProduct())
        right_part_2.add(Replicate(output_size, 1))
        right_part_2.add(MulConstant(-0.5))

        second_layer = ConcatTable()
        second_layer.add(left_part_2)
        second_layer.add(right_part_2)

        final_mlp = Sequential()
        final_mlp.add(first_layer)
        final_mlp.add(second_layer)
        final_mlp.add(CAddTable())

        return final_mlp
