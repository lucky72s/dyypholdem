
import settings.arguments as arguments

from nn.modules.module import Module


class MulConstant(Module):

    def __init__(self, constant_scalar):
        super(MulConstant, self).__init__()
        self.constant_scalar = constant_scalar

    def update_output(self, input):
        self.output = arguments.Tensor()
        self.output.resize_as_(input)

        self.output.copy_(input)
        self.output.mul_(self.constant_scalar)

        return self.output

    def update_grad_input(self, input, gradOutput):
        if self.gradInput is None:
            return

        self.gradInput.resize_as_(gradOutput)
        self.gradInput.copy_(gradOutput)
        self.gradInput.mul_(self.constant_scalar)

        return self.gradInput
