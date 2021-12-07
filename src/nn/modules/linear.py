import math

import settings.arguments as arguments

from nn.modules.module import Module
from nn.modules.utils import clear

import utils.pseudo_random as pseudo_random


class Linear(Module):

    def __init__(self, inputSize, outputSize, bias=True):
        super(Linear, self).__init__()
        self.weight = arguments.Tensor(outputSize, inputSize)
        self.gradWeight = arguments.Tensor(outputSize, inputSize)
        self.bias = arguments.Tensor(outputSize) if bias else None
        self.gradBias = arguments.Tensor(outputSize) if bias else None
        self.reset()

        self.addBuffer = None

    def update_output(self, input):
        assert input.dim() == 2

        self.output = arguments.Tensor(input.size(0), self.weight.size(0)).zero_()
        self._update_addBuffer(input)
        self.output.addmm_(input, self.weight.t(), alpha=1.0, beta=0.0)

        if self.bias is not None:
            self.output.addr_(self.addBuffer, self.bias)

        return self.output

    def update_grad_input(self, input, gradOutput):
        if self.gradInput is None:
            return

        nelement = self.gradInput.nelement()
        self.gradInput.resize_as_(input)
        if self.gradInput.nelement() != nelement:
            self.gradInput.zero_()

        assert input.dim() == 2
        self.gradInput.addmm_(gradOutput, self.weight, alpha=1.0, beta=0.0)

        return self.gradInput

    def acc_grad_parameters(self, input, gradOutput, scale=1):
        # serialization.serialize_as_tmp_t7("gradOutput-modern", gradOutput)
        assert input.dim() == 2
        # self.gradWeight.addmm_(scale, gradOutput.t(), input)        # deprecated
        self.gradWeight.addmm_(gradOutput.t(), input, alpha=scale)
        if self.bias is not None:
            # update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
            self._update_addBuffer(input)
            # self.gradBias.addmv_(scale, gradOutput.t(), self.addBuffer)     # depreacted
            self.gradBias.addmv_(gradOutput.t(), self.addBuffer, alpha=scale)

    def no_bias(self):
        self.bias = None
        self.gradBias = None
        return self

    def reset(self, stdv=None):
        arguments.logger.trace(f"Resetting 'Linear' module with size {repr(self.weight.size())}")
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.weight.size(1))

        if arguments.use_pseudo_random:
            pseudo_random.uniform_(self.weight, -stdv, stdv)
            if self.bias is not None:
                pseudo_random.uniform_(self.bias, -stdv, stdv)
        else:
            self.weight.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.uniform_(-stdv, stdv)
        return self

    def clear_state(self):
        clear(self, 'addBuffer')
        return super(Linear, self).clear_state()

    def _update_addBuffer(self, input):
        self.addBuffer = input.new(input.size(0)).fill_(1)

    def __repr__(self):
        return super(Linear, self).__repr__() + \
            '({} -> {})'.format(self.weight.size(1), self.weight.size(0)) + \
            (' without bias' if self.bias is None else '')
