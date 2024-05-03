import torch.nn.functional as F
from torch import Tensor


def bceloss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    pos_weight: Tensor | None = None,
    label_smoothing: float = 0.0
) -> Tensor:
    """
    Computes the binary cross entropy loss between `input` and `target`.

    This function is a wrapper around `torch.nn.functional.binary_cross_entropy_with_logits` with
    minor personal modifications. It supports label smoothing.

    Args:
        input (Tensor): The input tensor of shape (N, *), 
            where N is the batch size and the remaining dimensions represent 
            the input data.
        target (Tensor): The target tensor of shape (N, *), 
            where each value is expected to be either 0 or 1.
        weight (Tensor | None, optional): A manual rescaling weight given 
            to the loss of each batch element. Defaults to None.
        size_average (bool | None, optional): Deprecated (see `reduction`).
            By default, the losses are averaged over observations for each minibatch.
        reduce (bool | None, optional): Deprecated (see `reduction`).
            By default, the losses are averaged or summed for each minibatch.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Defaults to "mean".
        pos_weight (Tensor | None, optional): A weight of positive examples. 
            Must be a vector with length equal to the number of classes. 
            Defaults to None.
        label_smoothing (float, optional): A value in the range [0.0, 1.0] 
            that determines the amount of smoothing applied to the target labels. \
            Defaults to 0.0 (no smoothing).

    Returns:
        Tensor: The computed binary cross entropy loss.
    """
    assert 0 <= label_smoothing <= 1
    target = target * (1 - label_smoothing) + (1 - target) * label_smoothing

    return F.binary_cross_entropy_with_logits(
        input,
        target.float(),
        weight=weight,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        pos_weight=pos_weight,
    )


def cross_entropy_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average: bool = None,
    ignore_index = -100,
    reduce: bool = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0
) -> Tensor:
    """Computes the cross entropy loss between `input` and `target`.

    This function is a wrapper around `torch.nn.functional.cross_entropy` with 
    minor personal modifications.

    Args:
        input (Tensor): The input tensor of shape (N, C), 
            where N is the batch size and C is the number of classes.
        target (Tensor): The target tensor of shape (N, 1), 
            where each value is a class label in the range `[0, C-1]`.
        weight (Tensor | None, optional): A manual rescaling weight given to each class.
            If not provided, it defaults to `None`.
        size_average (bool, optional): Deprecated (see `reduction`).
            By default, the lossesare averaged over observations for each minibatch.
        ignore_index (int, optional): Specifies a target value that is ignored 
            and does not contribute to the input gradient. Default: `-100`.
        reduce (bool, optional): Deprecated (see `reduction`). By default, 
            the losses are averaged or summed for each minibatch.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'.
        label_smoothing (float, optional): A value in the range [0.0, 1.0] 
            that determines the amount of smoothing applied to the target labels. 
            Default: `0.0` (no smoothing).
 
    Returns:
        Tensor: The computed cross entropy loss.
    """

    target = target[:, 0].long()
    return F.cross_entropy(
        input=input,
        target=target,
        weight=weight,
        size_average=size_average,
        ignore_index=ignore_index,
        reduce=reduce,
        reduction=reduction,
        label_smoothing=label_smoothing
    )
