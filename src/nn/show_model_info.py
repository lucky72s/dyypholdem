import os
import sys
import argparse
sys.path.append(os.getcwd())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Model file to show information for")
    args = parser.parse_args()

    import settings.arguments as arguments
    from nn.value_nn import ValueNn

    nn = ValueNn().load_info_from_file(args.file)

    arguments.logger.info(repr(nn.model))
    arguments.logger.info(repr(nn))

    arguments.logger.success("Done.")

