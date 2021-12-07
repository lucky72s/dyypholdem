
import torch

import settings.arguments as arguments

from nn.modules.module import Module
from nn.modules.utils import clear


class DotProduct(Module):

    def __init__(self):
        super(DotProduct, self).__init__()
        self.gradInput = [arguments.Tensor(), arguments.Tensor()]
        self.buffer: torch.Tensor = None

    def update_output(self, input):
        input1, input2 = input[0], input[1]

        if self.buffer is None:
            self.buffer = input1.new()

        self.buffer = arguments.Tensor()
        self.buffer.resize_as_(input1)

        torch.mul(input1, input2, out=self.buffer)

        self.output = arguments.Tensor()
        self.output.resize_(self.buffer.size(0), 1)

        torch.sum(self.buffer, 1, True, out=self.output)
        self.output.resize_(input1.size(0))
        return self.output

    def update_grad_input(self, input, gradOutput):
        v1 = input[0]
        v2 = input[1]

        if len(self.gradInput) != 2:
            if self.gradInput[0] is None:
                self.gradInput[0] = input[0].new()
            if self.gradInput[1] is None:
                self.gradInput[1] = input[1].new()
            self.gradInput = self.gradInput[:2]

        gw1 = self.gradInput[0]
        gw2 = self.gradInput[1]
        gw1.resize_as_(v1).copy_(v2)
        gw2.resize_as_(v2).copy_(v1)

        go = gradOutput.contiguous().view(-1, 1).expand_as(v1)
        gw1.mul_(go)
        gw2.mul_(go)

        return self.gradInput

    def clear_state(self):
        clear(self, 'buffer')
        return super(DotProduct, self).clear_state()
