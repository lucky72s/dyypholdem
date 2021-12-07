import struct

import torch

import settings.arguments as arguments


def deserialize_from_torch7(file_name: str) -> torch.Tensor:

    ret: arguments.Tensor()

    with open(file_name, "r") as reader:
        input_line: str
        dimensions: int
        sizes: list
        strides: list
        element_count: int = 0
        elements: list
        total_size: int = 1
        offset: int = 0

        line_count = 0
        while True:
            input_line = reader.readline()
            if not input_line:
                break

            if input_line.startswith("torch.FloatTensor"):
                input_line = reader.readline()
                dimensions = int(input_line)
                input_line = reader.readline()
                str_sizes = input_line.split(' ')
                if len(str_sizes) > 0:
                    sizes = [0] * len(str_sizes)
                    for i in range(0, len(str_sizes)):
                        sizes[i] = int(str_sizes[i])
                        total_size *= sizes[i]

                input_line = reader.readline()
                str_strides = input_line.split(' ')
                if len(str_strides) > 0:
                    strides = [0] * len(str_strides)
                    for i in range(0, len(str_strides)):
                        strides[i] = int(str_strides[i])

                input_line = reader.readline()
                offset = int(input_line)
                offset -= 1

            if input_line.startswith("torch.FloatStorage"):
                input_line = reader.readline()
                element_count = int(input_line)
                if sizes:
                    input_line = reader.readline()
                    str_elements = input_line.split(' ')
                    if len(str_elements) > 0 and len(str_elements) == element_count:
                        ret = arguments.Tensor(sizes)
                        elements = [0.0] * total_size
                        for i in range(0, total_size):
                            elements[i] = float(str_elements[i + offset])
                        ret = arguments.Tensor(elements)
                        ret.resize_(sizes)
                        # ret.copy_(temp.view(ret.shape))

    return ret


def serialize_as_torch7(file_name: str, tensor: torch.Tensor, header=True, multi_line=False):

    source = tensor
    dimension_count = str(len(source.size()))
    total_size: int = 1
    sizes: str = ""
    strides: str = ""
    for i in range(0, len(source.size())):
        total_size *= source.size(i)
        sizes += str(source.size(i))
        sizes += " "
        strides += str(source.stride(i))
        strides += " "
    element_count = str(total_size)

    with open(file_name, "w") as writer:
        # header
        if header:
            writer.write("4" + '\n')
            writer.write("1" + '\n')
            writer.write("3" + '\n')
            writer.write("V 1" + '\n')
            writer.write("17" + '\n')
            writer.write("torch.FloatTensor" + '\n')
            writer.write(dimension_count + '\n')
            writer.write(sizes.strip() + '\n')
            writer.write(strides.strip() + '\n')
            writer.write(str(source.storage_offset() + 1) + '\n')
            writer.write("4" + '\n')
            writer.write("2" + '\n')
            writer.write("3" + '\n')
            writer.write("V 1" + '\n')
            writer.write("18" + '\n')
            writer.write("torch.FloatStorage" + '\n')
            writer.write(element_count + '\n')

        # elements
        source = source.flatten()

        # write as text-formatted floats
        elements_formatted = ["{:.9f}".format(elem) for elem in source.tolist()]
        if not multi_line:
            elements_clean = ' '.join(["0.00000000" if elem == "-0.00000000" else elem for elem in elements_formatted])
        else:
            elements_clean = '\n'.join(["0.00000000" if elem == "-0.00000000" else elem for elem in elements_formatted])
        writer.write(elements_clean)
        writer.write('\n')


def serialize_as_bin_torch7(file_name: str, tensor: torch.Tensor, header=True):

    source = tensor
    dimension_count = len(source.size())
    total_size: int = 1
    for i in range(0, len(source.size())):
        total_size *= source.size(i)

    with open(file_name, "wb") as writer:
        # header
        if header:
            writer.write(struct.pack("i", 4))
            writer.write(struct.pack("i", 1))
            writer.write(struct.pack("i", 3))
            writer.write("V 1".encode("ascii"))
            writer.write(struct.pack("i", 17))
            writer.write("torch.FloatTensor".encode("ascii"))
            writer.write(struct.pack("i", dimension_count))
            for i in range(dimension_count):
                writer.write(struct.pack("l", source.size(i)))
            for i in range(dimension_count):
                writer.write(struct.pack("l", source.stride(i)))
            writer.write(struct.pack("l", source.storage_offset() + 1))
            writer.write(struct.pack("i", 4))
            writer.write(struct.pack("i", 2))
            writer.write(struct.pack("i", 3))
            writer.write("V 1".encode("ascii"))
            writer.write(struct.pack("i", 18))
            writer.write("torch.FloatStorage".encode("ascii"))
            writer.write(struct.pack("l", total_size))

        # elements
        source = source.flatten()

        # write elements as bytes
        for i in range(total_size):
            writer.write(struct.pack('f', source.data[i]))
