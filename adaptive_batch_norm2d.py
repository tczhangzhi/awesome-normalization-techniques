import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveBatchNorm2d(nn.Module):
    """Adaptive Batch Normalization
    Parameters
        num_features – C from an expected input of size (N, C, H, W)
        eps – a value added to the denominator for numerical stability. Default: 1e-5
        momentum – the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
        affine – a boolean value that when set to True, this module has learnable affine parameters. Default: True

    Shape:
        Input: (N, C, H, W)
        Output: (N, C, H, W) (same shape as input)

    Examples:
        >>> m = AdaptiveBatchNorm2d(100)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)


if __name__ == '__main__':
    m = AdaptiveBatchNorm2d(100)
    input = torch.randn(20, 100, 35, 45)
    output = m(input)