
import settings.arguments as arguments

from nn.modules.module import Module


class Narrow(Module):

    def __init__(self, dimension, offset, length=1):
        super(Narrow, self).__init__()
        self.dimension = dimension
        self.index = offset
        self.length = length

    def update_output(self, input):
        length = self.length
        if length < 0:
            length = input.size(self.dimension) - self.index + self.length + 1

        output = input.narrow(self.dimension, self.index, length)

        self.output = arguments.Tensor()
        self.output.resize_as_(output).copy_(output)

        return self.output

    def update_grad_input(self, input, gradOutput):
        length = self.length
        if length < 0:
            length = input.size(self.dimension) - self.index + self.length + 1

        self.gradInput = self.gradInput.type_as(input)
        self.gradInput.resize_as_(input).zero_()
        self.gradInput.narrow(self.dimension, self.index, length).copy_(gradOutput)
        return self.gradInput
