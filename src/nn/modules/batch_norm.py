
import torch

import settings.arguments as arguments

from nn.modules.module import Module
from nn.modules.utils import clear

import utils.pseudo_random as pseudo_random


class BatchNormalization(Module):
    # expected dimension of input
    nDim = 2

    def __init__(self, nOutput, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNormalization, self).__init__()
        assert nOutput != 0

        self.affine = affine
        self.eps = eps
        self.train = True
        self.momentum = momentum
        self.running_mean = torch.zeros(nOutput).type(arguments.Tensor)
        self.running_var = torch.ones(nOutput).type(arguments.Tensor)

        self.save_mean: torch.Tensor = None
        self.save_std: torch.Tensor = None
        self.save_center: torch.Tensor = None
        self.save_norm: torch.Tensor = None
        self._input = None
        self._gradOutput = None

        if self.affine:
            self.weight = arguments.Tensor(nOutput)
            self.bias = arguments.Tensor(nOutput)
            self.gradWeight = arguments.Tensor(nOutput)
            self.gradBias = arguments.Tensor(nOutput)
            self.reset()
        else:
            self.weight = None
            self.bias = None
            self.gradWeight = None
            self.gradBias = None

    def update_output(self, input) -> torch.Tensor:
        self._check_input_dim(input)

        input = self._make_contiguous(input)[0]

        self.output = input.new(input.size(0), input.size(1))

        if self.save_mean is None:
            self.save_mean = input.new()
        self.save_mean.resize_as_(self.running_mean)
        if self.save_std is None:
            self.save_std = input.new()
        self.save_std.resize_as_(self.running_var)

        self.output = self._forward(input, self.weight, self.bias, self.train, self.momentum, self.eps)
        return self.output

    def update_grad_input(self, input, gradOutput):
        return self._backward(input, gradOutput, 1., self.gradInput)

    def backward(self, input, gradOutput, scale=1.):
        return self._backward(input, gradOutput, scale, self.gradInput, self.gradWeight, self.gradBias)

    def acc_grad_parameters(self, input, gradOutput, scale=1.):
        return self._backward(input, gradOutput, scale, None, self.gradWeight, self.gradBias)

    def _forward(self, input: torch.Tensor, weight: torch.Tensor, bias, train, momentum, eps):
        if train:
            sample_mean = torch.mean(input, 0)
            sample_var = torch.var(input, 0, unbiased=False)

            self.running_mean = momentum * sample_mean + (1 - momentum) * self.running_mean
            self.running_var = momentum * sample_var + (1 - momentum) * self.running_var

            self.save_std = torch.sqrt(sample_var + eps)
            self.save_mean = sample_mean
            self.save_center = input - sample_mean
            self.save_norm = self.save_center / self.save_std

            self.output = weight * self.save_norm + bias
        else:
            self.output = (input - self.running_mean) / torch.sqrt(self.running_var + eps)
            self.output = weight * self.output + bias
        return self.output

    def _backward(self, input, gradOutput, scale, gradInput=None, gradWeight=None, gradBias=None):
        self._check_input_dim(input)
        self._check_input_dim(gradOutput)
        if not hasattr(self, 'save_mean') or not hasattr(self, 'save_std'):
            raise RuntimeError('you have to call updateOutput() at least once before backward()')

        input, gradOutput = self._make_contiguous(input, gradOutput)
        N = gradOutput.size(0)

        scale = scale or 1.
        if gradInput is not None:
            gradInput.resize_as_(gradOutput).zero_()

        gradWeight.copy_(torch.sum(gradOutput * self.save_norm, 0))
        gradBias.copy_(torch.sum(gradOutput, 0))

        dx_norm = gradOutput * self.weight
        self.gradInput = 1 / N / self.save_std * (
                N * dx_norm - torch.sum(dx_norm, 0) - self.save_norm * torch.sum(dx_norm * self.save_norm, 0))
        return self.gradInput

    def reset(self):
        arguments.logger.trace(f"Resetting 'BatchNormalization' module with size {repr(self.weight.size())}")

        if arguments.use_pseudo_random:
            if self.weight is not None:
                pseudo_random.uniform_(self.weight, 0.0, 1.0)
        else:
            if self.weight is not None:
                self.weight.uniform_()

        if self.bias is not None:
            self.bias.zero_()

        self.running_mean.zero_()
        self.running_var.fill_(1)

    def clear_state(self):
        # first 5 buffers are not present in the current implementation,
        # but we keep them for cleaning old saved models
        clear(self, [
            'buffer',
            'buffer2',
            'centered',
            'std',
            'normalized',
            '_input',
            '_gradOutput',
            'save_mean',
            'save_std',
        ])
        return super(BatchNormalization, self).clear_state()

    def _check_input_dim(self, input):
        if input.dim() != self.nDim:
            raise RuntimeError(
                'only mini-batch supported ({}D tensor), got {}D tensor instead'.format(self.nDim, input.dim()))
        if input.size(1) != self.running_mean.nelement():
            raise RuntimeError('got {}-feature tensor, expected {}'.format(input.size(1), self.running_mean.nelement()))

    def _make_contiguous(self, input, gradOutput=None):
        if not input.is_contiguous():
            if self._input is None:
                self._input = input.new()
            self._input.resize_as_(input).copy_(input)
            input = self._input

        if gradOutput is not None:
            if not gradOutput.is_contiguous():
                if self._gradOutput is None:
                    self._gradOutput = gradOutput.new()
                self._gradOutput.resize_as_(gradOutput).copy_(gradOutput)
                gradOutput = self._gradOutput

        return input, gradOutput
