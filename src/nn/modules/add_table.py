
import settings.arguments as arguments

from nn.modules.module import Module


class CAddTable(Module):

    def __init__(self):
        super(CAddTable, self).__init__()
        self.gradInput = []

    def update_output(self, input):
        self.output = arguments.Tensor()
        self.output.resize_as_(input[0]).copy_(input[0])

        for i in range(1, len(input)):
            self.output.add_(input[i])

        return self.output

    def update_grad_input(self, input, gradOutput):
        for i in range(len(input)):
            if i >= len(self.gradInput):
                assert i == len(self.gradInput)
                self.gradInput.append(input[0].new())
            self.gradInput[i].resize_as_(input[i]).copy_(gradOutput)

        del self.gradInput[len(input):]

        return self.gradInput
