import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainSpecificBatchNorm(nn.Module):
    """Domain-Specific Batch Normalization
        Parameters
            num_features – C from an expected input of size (N, C, H, W)
            eps – a value added to the denominator for numerical stability. Default: 1e-5
            momentum – the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
            affine – a boolean value that when set to True, this module has learnable affine parameters. Default: True

        Shape:
            Input: (N, C, H, W) (N,)
            Output: (N, C, H, W) (same shape as input)

        Examples:
            >>> m = DomainSpecificBatchNorm(100, 10)
            >>> input = torch.randn(20, 100, 35, 45)
            >>> label = torch.randint(10, (20,))
            >>> output = m(input, label)
    """
    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(DomainSpecificBatchNorm, self).__init__()
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, y):
        bn = self.bns[y[0]]
        return bn(x)


if __name__ == '__main__':
    m = DomainSpecificBatchNorm(100, 10)
    input = torch.randn(20, 100, 35, 45)
    label = torch.randint(10, (20, ))
    output = m(input, label)