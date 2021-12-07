import struct
from collections import OrderedDict

from nn.modules import module_factory
from torch7 import torch7_factory

TYPE_NIL = 0
TYPE_NUMBER = 1
TYPE_STRING = 2
TYPE_TABLE = 3
TYPE_TORCH = 4
TYPE_BOOLEAN = 5
TYPE_FUNCTION = 6
TYPE_RECUR_FUNCTION = 8
LEGACY_TYPE_RECUR_FUNCTION = 7

module_factory = module_factory.ModuleFactory()
torch7_factory = torch7_factory.TorchFactory()


class Torch7File(object):

    def __init__(self, file_obj):
        self.file = file_obj
        self.file_objects = {}
        self.index = 0

    def read_torch7_object(self):
        if 'b' in self.file.mode:
            return self.read_torch7_object_binary()
        else:
            return self.read_torch7_object_ascii()

    def read_torch7_object_ascii(self):
        type_line = self.file.readline().strip()
        if type_line:
            typeidx = int(type_line)

            if typeidx == TYPE_NUMBER:
                return float(self.file.readline().strip())
            elif typeidx == TYPE_BOOLEAN:
                return bool(self.file.readline().strip())
            elif typeidx == TYPE_STRING:
                size = int(self.file.readline().strip())
                return str(self.file.read(size + 1)).strip()
            elif typeidx == TYPE_FUNCTION:
                raise NotImplementedError()
            elif typeidx in {TYPE_TABLE, TYPE_TORCH, TYPE_RECUR_FUNCTION, LEGACY_TYPE_RECUR_FUNCTION}:
                self.index = int(self.file.readline())
                if self.index in self.file_objects:
                    return self.file_objects[self.index]

                if typeidx in {TYPE_RECUR_FUNCTION, LEGACY_TYPE_RECUR_FUNCTION}:
                    raise NotImplementedError()
                elif typeidx == TYPE_TORCH:
                    version = str(self.file.read(int(self.file.readline())+1)).strip()         # only accept version V 1 or later
                    class_name = str(self.file.read(int(self.file.readline())+1)).strip()
                    if class_name.startswith("nn"):
                        torch_object = module_factory.create_module(class_name)
                    elif class_name == "torch.FloatTensor" or class_name == "torch.CudaTensor":
                        torch_object = torch7_factory.create_torch_object(class_name)
                    else:
                        raise TypeError()
                    self.file_objects[self.index] = torch_object

                    if torch_object is not None:
                        if hasattr(torch_object, "read"):
                            torch_object.__getattribute__("read")(self)
                        else:
                            raise NameError()

                    return torch_object
                else:       # it's a torch table object
                    size = int(self.file.readline().strip())
                    torch_object = OrderedDict()
                    self.file_objects[self.index] = torch_object

                    for i in range(size):
                        k = self.read_torch7_object()
                        v = self.read_torch7_object()
                        torch_object[k] = v

                    return torch_object
            else:
                raise TypeError()
        else:
            # reached EOF
            return

    def read_torch7_object_binary(self):
        type_line = self.file.read(4)
        if type_line:
            typeidx = struct.unpack('i', type_line)[0]

            if typeidx == TYPE_NUMBER:
                return float(struct.unpack('d', self.file.read(8))[0])
            elif typeidx == TYPE_BOOLEAN:
                return bool(struct.unpack('i', self.file.read(4))[0])
            elif typeidx == TYPE_STRING:
                size = int(struct.unpack('i', self.file.read(4))[0])
                return str(self.file.read(size), 'utf-8')
            elif typeidx == TYPE_FUNCTION:
                raise NotImplementedError()
            elif typeidx in {TYPE_TABLE, TYPE_TORCH, TYPE_RECUR_FUNCTION, LEGACY_TYPE_RECUR_FUNCTION}:
                self.index = int(struct.unpack('i', self.file.read(4))[0])
                if self.index in self.file_objects:
                    return self.file_objects[self.index]

                if typeidx in {TYPE_RECUR_FUNCTION, LEGACY_TYPE_RECUR_FUNCTION}:
                    raise NotImplementedError()
                elif typeidx == TYPE_TORCH:
                    version = str(self.file.read(int(struct.unpack('i', self.file.read(4))[0])), 'utf-8')  # only accept version V 1 or later
                    class_name = str(self.file.read(int(struct.unpack('i', self.file.read(4))[0])), 'utf-8')
                    if class_name.startswith("nn"):
                        torch_object = module_factory.create_module(class_name)
                    elif class_name == "torch.FloatTensor" or class_name == "torch.CudaTensor":
                        torch_object = torch7_factory.create_torch_object(class_name)
                    else:
                        raise TypeError()
                    self.file_objects[self.index] = torch_object

                    if torch_object is not None:
                        if hasattr(torch_object, "read"):
                            torch_object.__getattribute__("read")(self)
                        else:
                            raise NameError()

                    return torch_object
                else:  # it's a torch table object
                    size = int(struct.unpack('i', self.file.read(4))[0])
                    torch_object = OrderedDict()
                    self.file_objects[self.index] = torch_object

                    for i in range(size):
                        k = self.read_torch7_object()
                        v = self.read_torch7_object()
                        torch_object[k] = v

                    return torch_object
            else:
                raise TypeError()
        else:
            # reached EOF
            return


def read_model_from_torch7_file(file_name, mode="r"):
    with open(file_name, mode) as file_obj:
        src_file = Torch7File(file_obj)
        return src_file.read_torch7_object()


