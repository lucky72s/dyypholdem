import os
import sys
import argparse
sys.path.append(os.getcwd())


def convert_t7_to_pytorch(file_name: str, street: int, mode="ascii"):

    torch7_info = torch7_file.read_model_from_torch7_file(file_name.replace(".model", ".info"), mode)
    torch7_model = torch7_file.read_model_from_torch7_file(file_name, mode)
    if arguments.use_gpu:
        torch7_model.cuda()

    model_file_name = file_name.replace(".model", ".tar")
    ValueNn().save_model(torch7_model, model_file_name, street=street, epoch=int(torch7_info['epoch']), valid_loss=torch7_info['valid_loss'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="torch7 model file to convert")
    parser.add_argument("street", type=int, choices=[1, 2, 3, 4], help="the street of the model (1=pre-flop, 2=flop, 3=turn, 4=river)")
    parser.add_argument("mode", type=str, choices=['binary', 'ascii'], help="file mode of the model ('binary' or 'ascii'")
    args = parser.parse_args()

    import settings.arguments as arguments

    from nn.value_nn import ValueNn
    import torch7_file as torch7_file

    file_mode = "rb"
    if args.mode == "ascii":
        file_mode = "r"

    arguments.logger.info(f"Converting file '{args.file}' with type '{args.mode}' for target device '{arguments.device}'")
    convert_t7_to_pytorch(args.file, args.street, file_mode)
    arguments.logger.success("Conversion completed")
