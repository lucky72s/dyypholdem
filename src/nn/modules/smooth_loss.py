
import torch
import torch.nn.functional as F

from nn.modules.criterion import Criterion


class SmoothL1Criterion(Criterion):

    def __init__(self, size_average=True, reduce=None, reduction: str = 'mean', beta: float = 1.0):
        super(SmoothL1Criterion, self).__init__()
        self.sizeAverage = size_average
        self.output_tensor = None
        self.reduction = reduction
        self.beta = beta

    def update_output(self, input, target):
        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        self.output_tensor = F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)
        self.output = self.output_tensor.item()
        return self.output

    def update_grad_input(self, input, target):
        self.gradInput = input.new(input.size())
        norm = 1.0 / input.numel() if self.sizeAverage else 1.0

        delta = input.data - target.data
        self.gradInput.copy_(delta)
        self.gradInput *= norm
        self.gradInput[torch.lt(delta, -1.0)] = -norm
        self.gradInput[torch.gt(delta, 1.0)] = norm

        return self.gradInput
