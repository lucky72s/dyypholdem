import os
import sys
import argparse
sys.path.append(os.getcwd())


def train(street: int):

    start_epoch = 0
    state = None
    if arguments.resume_training:
        # reload existing network
        nn = ValueNn().load_for_street(street - 1, training=True)
        start_epoch = nn.model_info["epoch"]
        model = nn.model
        state = nn.model_state
    else:
        # create empty neural network
        model = TrainingNetwork().build_net(street)

    data_stream = DataStream(street)

    training.train(street, model, state, data_stream, start_epoch, start_epoch + arguments.epoch_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network for the specified street')
    parser.add_argument('street', type=int, choices=[1, 2, 3, 4], help="Street (1=pre-flop, 2=flop, 3=turn, 4=river)")
    args = parser.parse_args()

    import settings.arguments as arguments

    from nn.net_builder import TrainingNetwork
    from training.data_stream import DataStream
    import training.train as training
    from nn.value_nn import ValueNn

    street_arg = int(sys.argv[1])

    arguments.logger.info(f"Training model for street: {street_arg}")

    train(street_arg)
