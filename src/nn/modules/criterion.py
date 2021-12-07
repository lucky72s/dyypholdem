
import settings.arguments as arguments

from nn.modules.utils import recursive_type


class Criterion(object):

    def __init__(self):
        self.gradInput = arguments.Tensor()
        self.output = 0

    def forward(self, input, target):
        return self.update_output(input, target)

    def update_output(self, input, target):
        raise NotImplementedError

    def backward(self, input, target):
        return self.update_grad_input(input, target)

    def update_grad_input(self, input, target):
        raise NotImplementedError

    def type(self, type, tensorCache=None):
        # find all tensors and convert them
        for key, param in self.__dict__.items():
            setattr(self, key, recursive_type(param, type, tensorCache or {}))
        return self

    def float(self):
        return self.type('torch.FloatTensor')

    def double(self):
        return self.type('torch.DoubleTensor')

    def cuda(self):
        return self.type('torch.cuda.FloatTensor')
