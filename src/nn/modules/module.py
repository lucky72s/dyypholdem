from collections import OrderedDict

import torch

import settings.arguments as arguments

from nn.modules.utils import clear, recursive_type, recursive_fill


class Module(object):

    train: bool

    def __init__(self):
        self.gradInput = arguments.Tensor()
        self.output = arguments.Tensor()
        self._type = self.output.type()

    def __repr__(self):
        return 'nn.' + self.__class__.__name__

    def parameters(self):
        has_weight = hasattr(self, 'weight') and self.weight is not None
        has_bias = hasattr(self, 'bias') and self.bias is not None
        if has_weight and has_bias:
            return [self.weight, self.bias], [self.gradWeight, self.gradBias]
        elif has_weight:
            return [self.weight], [self.gradWeight]
        elif has_bias:
            return [self.bias], [self.gradBias]
        else:
            return

    def forward(self, input):
        return self.update_output(input)

    def update_output(self, input):
        return self.output

    def backward(self, input, gradOutput, scale=1):
        self.update_grad_input(input, gradOutput)
        self.acc_grad_parameters(input, gradOutput, scale)
        return self.gradInput

    def update_grad_input(self, input, gradOutput):
        return self.gradInput

    def acc_grad_parameters(self, input, gradOutput, scale=1):
        pass

    def zero_grad_parameters(self):
        params = self.parameters()
        if params is not None:
            for grad in params[1]:
                grad.zero_()

    def update_parameters(self, learningRate):
        if self.parameters() is not None:
            params, gradParams = self.parameters()
            if params:
                for p, gp in zip(params, gradParams):
                    p.add_(-learningRate, gp)

    def training(self):
        self.train = True

    def evaluate(self):
        self.train = False

    def apply(self, callback):
        callback(self)
        if hasattr(self, 'modules'):
            for module in self.modules:
                module.apply(callback)

    def type(self, type=None, tensorCache=None):
        if type is None:
            return self._type

        tensorCache = tensorCache or {}

        # find all tensors and convert them
        for key, param in self.__dict__.items():
            setattr(self, key, recursive_type(param, type, tensorCache))

        self._type = type
        return self

    def float(self, *args):
        return self.type('torch.FloatTensor', *args)

    def double(self, *args):
        return self.type('torch.DoubleTensor', *args)

    def cpu(self, *args):
        if not arguments.use_gpu:
            return self.type(arguments.Tensor, *args)
        else:
            return self

    def cuda(self, *args):
        if arguments.use_gpu:
            return self.type(arguments.Tensor, *args)
        else:
            return self

    def reset(self):
        pass

    def clear_state(self):
        return clear(self, 'output', 'gradInput')

    def fill(self, val):
        for key, param in self.__dict__.items():
            recursive_fill(param, val)

    def read(self, torch_file):
        module: dict = torch_file.read_torch7_object()
        for k, v in module.items():
            if k in {"index", "dimension", "dim", "ndim"}:
                v = int(v - 1)
            if k in {"length", "nf", "nOutputPlane", "nfeatures"}:
                v = int(v)
            if type(v) is OrderedDict:
                list_values = list(v.values())
                self.__dict__[k] = list_values
            else:
                self.__dict__[k] = v

    # This function is not easy to understand. It works as follows:
    #
    # - gather all parameter tensors for this module (and children);
    #   count all parameter values (floats)
    # - create one ginormous memory area (Storage object) with room for all
    #   parameters
    # - remap each parameter tensor to point to an area within the ginormous
    #   Storage, and copy it there
    #
    # It has the effect of making all parameters point to the same memory area,
    # which is: returned.
    #
    # The purpose is to allow operations over all parameters (such as momentum
    # updates and serialization), but it assumes that all parameters are of
    # the same type (and, in the case of CUDA, on the same device), which
    # is not always True. Use for_each() to iterate over this module and
    # children instead.
    #
    # Module._flattenTensorBuffer can be used by other packages (e.g. cunn)
    # to specify the type of temporary buffers. For example, the temporary
    # buffers for CudaTensor could be FloatTensor, to avoid GPU memory usage.
    #
    _flatten_tensor_buffer = {}

    def flatten_parameters(self):
        _params = self.parameters()
        if _params is None:
            return
        parameters, grad_parameters = _params
        p, g = self._flatten(parameters), self._flatten(grad_parameters)

        assert p.nelement() == g.nelement()
        if parameters:
            for param, grad in zip(parameters, grad_parameters):
                assert param.storage_offset() == grad.storage_offset()
        return p, g

    @staticmethod
    def _flatten(parameters=[]):

        # returns True if tensor occupies a contiguous region of memory (no holes)
        def is_compact(tensor):
            # isn't it enough to check if strides == size.cumprod(0)?
            sorted_stride, perm = torch.sort(torch.LongTensor(tensor.stride()), 0, True)
            sorted_size = torch.LongTensor(list(tensor.size())).index_select(0, perm)
            n_real_dim = int(torch.clamp(sorted_stride, 0, 1).sum())
            sorted_stride = sorted_stride.narrow(0, 0, n_real_dim).clone()
            sorted_size = sorted_size.narrow(0, 0, n_real_dim).clone()
            t = tensor.new().set_(tensor.storage(), 0,
                                  tuple(sorted_size),
                                  tuple(sorted_stride))
            return t.is_contiguous()

        if not parameters:
            return torch.Tensor()

        temp_tensor = parameters[0].new
        buffer_tensor = Module._flatten_tensor_buffer.get(type(parameters[0]), temp_tensor)

        # 1. construct the set of all unique storages referenced by parameter tensors
        storages = {}
        num_parameters = 0
        parameter_meta = []
        for i, param in enumerate(parameters):
            storage = param.storage()
            key = storage._cdata

            if key not in storages:
                storages[key] = (storage, num_parameters)
                num_parameters = num_parameters + storage.size()

            parameter_meta.append({
                'storage_offset': param.storage_offset() + storages[key][1],
                'size': param.size(),
                'stride': param.stride()
            })

        # 2. construct a single tensor that will hold all the parameters
        flat_parameters = buffer_tensor(num_parameters).zero_()

        # 3. determine if there are elements in the storage that none of the
        #    parameter tensors reference ('holes')
        tensors_compact = True
        for meta in parameter_meta:
            tmp = buffer_tensor().set_(flat_parameters.storage(), meta['storage_offset'], meta['size'], meta['stride'])
            tmp.fill_(1)
            tensors_compact = tensors_compact and is_compact(tmp)

        mask_parameters = flat_parameters.byte().clone()
        compact_offsets = flat_parameters.long().cumsum(0)
        used_parameters = compact_offsets[-1]

        # 4. copy storages into the flattened parameter tensor
        for storageAndOffset in storages.values():
            storage, offset = storageAndOffset
            flat_parameters[slice(offset, offset + storage.size())].copy_(temp_tensor().set_(storage))

        # 5. allow garbage collection
        storages = None
        for param in parameters:
            param.set_()

        # 6. compact the flattened parameters if there were holes
        if used_parameters != num_parameters:
            assert tensors_compact

            flat_parameters = buffer_tensor(used_parameters).copy_(flat_parameters.masked_select(mask_parameters))
            for meta in parameter_meta:
                meta['storage_offset'] = compact_offsets[meta['storage_offset']]

        if buffer_tensor != temp_tensor:
            flat_parameters = temp_tensor(flat_parameters.nelement()).copy_(flat_parameters)

        # 7. fix up the parameter tensors to point at the flattened parameters
        for param, meta in zip(parameters, parameter_meta):
            param.set_(flat_parameters.storage(),
                       meta['storage_offset'],
                       meta['size'],
                       meta['stride'])

        return flat_parameters

