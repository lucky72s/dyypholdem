import struct

import torch

import settings.arguments as arguments


class Torch7Tensor(torch.Tensor):
    torch_type: str
    dimensions: int
    sizes: list
    strides: list
    elements: list
    total_size: int = 1
    offset: int = 0

    def __new__(cls):
        if arguments.use_gpu:
            return super().__new__(cls).cuda()
        else:
            return super().__new__(cls)

    def __init__(self):
        super().__init__()

    def read(self, torch_file):
        if 'b' in torch_file.file.mode:
            self.read_binary(torch_file)
        else:
            self.read_ascii(torch_file)

    def read_ascii(self, torch_file):
        self.dimensions = int(torch_file.file.readline().strip())
        if self.dimensions != 0:
            str_sizes = torch_file.file.readline().split(' ')
            if len(str_sizes) > 0:
                self.sizes = [0] * len(str_sizes)
                for i in range(0, len(str_sizes)):
                    self.sizes[i] = int(str_sizes[i])
                    self.total_size *= self.sizes[i]
            str_strides = torch_file.file.readline().split(' ')
            if len(str_strides) > 0:
                self.strides = [0] * len(str_strides)
                for i in range(0, len(str_strides)):
                    self.strides[i] = int(str_strides[i])
            self.offset = int(torch_file.file.readline().strip())
            self.offset -= 1

            """ assuming storage follows directly in file or is referenced """
            type_idx = int(torch_file.file.readline().strip())
            idx = int(torch_file.file.readline().strip())
            if idx in torch_file.file_objects:
                ref: Torch7Tensor = torch_file.file_objects[idx]
                self.set_(ref.storage(), self.offset, torch.Size(self.sizes), tuple(self.strides))
            else:
                # skip over version
                torch_file.file.readline()
                torch_file.file.readline()
                torch_file.file.readline()
                str_storage = torch_file.file.readline().strip()
                if str_storage == "torch.FloatStorage" or str_storage == "torch.CudaStorage":
                    element_count = int(torch_file.file.readline().strip())
                    str_storage = torch_file.file.readline().split(' ')
                    storage = [0.0] * element_count
                    for i in range(0, element_count):
                        storage[i] = float(str_storage[i])

                    temp = arguments.Tensor(storage)
                    self.resize_(element_count)
                    self.copy_(temp)
                    self.as_strided_(self.sizes, self.strides, self.offset)

                    torch_file.file_objects[idx] = self
        else:
            self.resize_(0)
            torch_file.file.readline()
            torch_file.file.readline()

    def read_binary(self, torch_file):
        self.dimensions = int(struct.unpack('i', torch_file.file.read(4))[0])
        if self.dimensions != 0:
            self.sizes = [0] * self.dimensions
            for i in range(0, self.dimensions):
                self.sizes[i] = int(struct.unpack('q', torch_file.file.read(8))[0])
                self.total_size *= self.sizes[i]
            self.strides = [0] * self.dimensions
            for i in range(0, self.dimensions):
                self.strides[i] = int(struct.unpack('q', torch_file.file.read(8))[0])
            self.offset = int(struct.unpack('q', torch_file.file.read(8))[0])
            self.offset -= 1

            """ assuming storage follows directly in file or is referenced """
            type_idx = int(struct.unpack('i', torch_file.file.read(4))[0])
            idx = int(struct.unpack('i', torch_file.file.read(4))[0])
            if idx in torch_file.file_objects:
                ref: Torch7Tensor = torch_file.file_objects[idx]
                self.set_(ref.storage(), self.offset, torch.Size(self.sizes), tuple(self.strides))
            else:
                # skip over version
                version_size = int(struct.unpack('i', torch_file.file.read(4))[0])
                torch_file.file.read(version_size)
                size = int(struct.unpack('i', torch_file.file.read(4))[0])
                str_storage = str(torch_file.file.read(size), 'utf-8')
                if str_storage == "torch.FloatStorage" or str_storage == "torch.CudaStorage":
                    element_count = int(struct.unpack('q', torch_file.file.read(8))[0])
                    byte_arr = torch_file.file.read(element_count*4)
                    storage = torch.FloatStorage.from_buffer(byte_arr, byte_order="native")
                    if arguments.use_gpu:
                        storage = storage.cuda()
                    self.set_(storage, self.offset, torch.Size(self.sizes), tuple(self.strides))
                    torch_file.file_objects[idx] = self
        else:
            self.resize_(0)
            torch_file.file.read(12)


class FloatTensor(Torch7Tensor):

    def __init__(self):
        super().__init__()
        self.torch_type = "torch.FloatTensor"


class CudaTensor(Torch7Tensor):

    def __init__(self):
        super().__init__()
        self.torch_type = "torch.CudaTensor"


class TorchFactory(object):

    torch_types = {
        "torch.FloatTensor": FloatTensor,
        "torch.CudaTensor": CudaTensor,
    }

    def create_torch_object(self, torch_obj) -> torch.Tensor:
        return self.torch_types[torch_obj]()
