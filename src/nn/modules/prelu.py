
import torch

import settings.arguments as arguments

from nn.modules.module import Module
from nn.modules.utils import clear


class PReLU(Module):

    def __init__(self, nOutputPlane=0):
        super(PReLU, self).__init__()
        # if no argument provided, use shared model (weight is scalar)
        self.nOutputPlane = nOutputPlane
        self.weight = arguments.Tensor(nOutputPlane or 1).fill_(0.25)
        self.gradWeight = arguments.Tensor(nOutputPlane or 1)

    def update_output(self, input):
        self.output.resize_as_(input)
        self.output.copy_(input)
        self.output[torch.le(input, 0.0)] *= self.weight
        return self.output

    def update_grad_input(self, input, gradOutput):
        self.gradInput.resize_as_(gradOutput)
        self.gradInput.copy_(gradOutput)
        self.gradInput[torch.le(input, 0.0)] *= self.weight
        return self.gradInput

    def acc_grad_parameters(self, input, gradOutput, scale=1):
        idx = torch.le(input, 0)
        _sum = torch.sum(input[idx] * gradOutput[idx])
        self.gradWeight += scale * _sum
        return self.gradWeight

    def clear_state(self):
        clear(self, 'gradWeightBuf', 'gradWeightBuf2')
        return super(PReLU, self).clear_state()
