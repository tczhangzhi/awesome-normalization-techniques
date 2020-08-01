import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceEnhancementBatchNorm2d(nn.BatchNorm2d):
    """Instance Enhancement Batch Normalization
        Parameters
            num_features – C from an expected input of size (N, C, H, W)
            eps – a value added to the denominator for numerical stability. Default: 1e-5
            momentum – the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
            affine – a boolean value that when set to True, this module has learnable affine parameters. Default: True

        Shape:
            Input: (N, C, H, W)
            Output: (N, C, H, W) (same shape as input)

        Examples:
            >>> m = InstanceEnhancementBatchNorm2d(100)
            >>> input = torch.randn(20, 100, 35, 45)
            >>> output = m(input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(InstanceEnhancementBatchNorm2d, self).__init__(num_features, eps, momentum, affine)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias_readjust = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust.data.fill_(0)
        self.bias_readjust.data.fill_(-1)
        self.weight.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, input):
        self._check_input_dim(input)

        attention = self.sigmoid(self.avg(input) * self.weight_readjust + self.bias_readjust)
        bn_w = self.weight * attention

        out_bn = F.batch_norm(input, self.running_mean, self.running_var, None, None, self.training, self.momentum,
                              self.eps)
        out_bn = out_bn * bn_w + self.bias

        return out_bn


if __name__ == '__main__':
    m = InstanceEnhancementBatchNorm2d(100)
    input = torch.randn(20, 100, 35, 45)
    output = m(input)