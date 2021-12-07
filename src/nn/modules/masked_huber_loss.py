
import torch

import settings.arguments as arguments

from nn.modules.criterion import Criterion
from nn.modules.smooth_loss import SmoothL1Criterion


class MaskedHuberLoss(Criterion):

    mask_sum: torch.Tensor

    def __init__(self):
        super().__init__()
        self.criterion = SmoothL1Criterion()
        self.mask_sum = None
        self.mask_placeholder = None
        self.mask_multiplier = None

    # --- Computes the loss over a batch of neural net outputs and targets.
    # --
    # -- @param outputs an NxM tensor containing N vectors of values over buckets,
    # -- output by the neural net
    # -- @param targets an NxM tensor containing N vectors of actual values over
    # -- buckets, produced by @{data_generation_call}
    # -- @param mask an NxM tensor containing N mask vectors generated with
    # -- @{bucket_conversion.get_possible_bucket_mask}
    # -- @return the sum of Huber loss applied elementwise on `outputs` and `targets`,
    # -- masked so that only valid buckets are included
    def forward(self, outputs, targets, mask=None):

        batch_size = outputs.size(0)
        feature_size = outputs.size(1)

        # 1.0 zero out the outputs/target so that the error does not depend on these
        outputs.mul_(mask)
        targets.mul(mask)

        loss = self.criterion.forward(outputs, targets)

        # 2.0 if the batch size has changed, create new storage for the sum, otherwise reuse
        if self.mask_sum is None or (self.mask_sum.size(0) != batch_size):
            self.mask_placeholder = arguments.Tensor(mask.size()).fill_(0)
            self.mask_sum = arguments.Tensor(batch_size).fill_(0)
            self.mask_multiplier = self.mask_sum.clone().fill_(0).view(-1, 1)

        # 3.0 compute mask sum for each batch
        self.mask_placeholder.copy_(mask)
        self.mask_sum = torch.sum(self.mask_placeholder, 1)

        # 3.1 mask multiplier - note that mask is 1 for impossible features
        self.mask_multiplier.fill_(feature_size)
        self.mask_multiplier.div_(self.mask_sum.view(self.mask_multiplier.shape))

        # 4.0 multiply to get a new loss
        # loss is not really computed batch-wise correctly,
        # but that does not really matter now since gradients are correct
        loss_multiplier = (batch_size * feature_size) / self.mask_sum.sum()
        new_loss = loss_multiplier * loss

        return new_loss

    # --- Computes the gradient of the loss function @{forward} with
    # -- arguments `outputs`, `targets`, and `mask`.
    # --
    # -- Must be called after a @{forward} call with the same arguments.
    # --
    # -- @param outputs an NxM tensor containing N vectors of values over buckets,
    # -- output by the neural net
    # -- @param targets an NxM tensor containing N vectors of actual values over
    # -- buckets, produced by @{data_generation_call}
    # -- @param mask an NxM tensor containing N mask vectors generated with
    # -- @{bucket_conversion.get_possible_bucket_mask}
    # -- @return the gradient of @{forward} applied to the arguments
    def backward(self, outputs, targets):
        dloss_doutput = self.criterion.backward(outputs, targets)
        # we use the multiplier computed with the mask during forward call
        dloss_doutput.mul_(self.mask_multiplier.expand_as(dloss_doutput))

        return dloss_doutput
