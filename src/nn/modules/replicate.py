
import torch

from nn.modules.module import Module


class Replicate(Module):

    def __init__(self, nf, dim=0, ndim=0):
        super(Replicate, self).__init__()
        self.nfeatures = nf
        self.dim = dim
        self.ndim = ndim
        assert self.dim >= 0

    def update_output(self, input):
        assert self.dim <= input.dim()

        size = [int(x) for x in input.size()]           # convert sizes to int

        size.insert(int(self.dim), int(self.nfeatures))

        stride = list(input.stride())
        stride.insert(self.dim, 0)

        self.output.set_(input.storage(), input.storage_offset(),
                         torch.Size(size), tuple(stride))
        return self.output

    def update_grad_input(self, input, gradOutput):
        self.gradInput.resize_as_(input).zero_()
        size = list(input.size())
        size.insert(self.dim, 1)

        grad_input = self.gradInput.view(*size)
        torch.sum(gradOutput, self.dim, True, out=grad_input)
        return self.gradInput
