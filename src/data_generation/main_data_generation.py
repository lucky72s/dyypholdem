import os
import sys
import argparse
sys.path.append(os.getcwd())


def run(street: int):
    if street == 0:
        data_generation = DataGeneratorAux()
        data_generation.generate_data(arguments.gen_data_count, 1)
    else:
        data_generation = DataGenerator()
        data_generation.generate_data(arguments.gen_data_count, street)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data for the specified street')
    parser.add_argument('street', type=int, choices=[0, 1, 2, 3, 4], help="Street (0=preflop-aux, 1=pre-flop, 2=flop, 3=turn, 4=river)")
    args = parser.parse_args()

    import settings.arguments as arguments
    from data_generation import DataGenerator
    from aux_data_generation import DataGeneratorAux

    street_arg = int(sys.argv[1])
    arguments.logger.info(f"Generating data for street: {street_arg}")
    run(street_arg)
