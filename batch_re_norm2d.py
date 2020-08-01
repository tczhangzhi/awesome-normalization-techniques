import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchReNorm2D(nn.Module):
    """Batch Re-Normalization
    Parameters
        num_features – C from an expected input of size (N, C, H, W)
        eps – a value added to the denominator for numerical stability. Default: 1e-5
        momentum – the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
        affine – a boolean value that when set to True, this module has learnable affine parameters. Default: True
        r_max - a hyper parameter. The paper used rmax = 1 for the first 5000 training steps, after which these were gradually relaxed to reach rmax=3 at 40k steps.
        d_max - a hyper parameter. The paper used dmax = 0 for the first 5000 training steps, after which these were gradually relaxed to reach dmax=5 at 25k steps.

    Shape:
        Input: (N, C, H, W)
        Output: (N, C, H, W) (same shape as input)

    Examples:
        >>> m = BatchReNorm2d(100)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """
    def __init__(self, num_features, r_max=1, d_max=0, eps=1e-3, momentum=0.01, affine=True):

        super(BatchReNorm2D, self).__init__()
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))

        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))

        self.r_max, self.d_max = r_max, d_max
        self.eps, self.momentum = eps, momentum

    def update_stats(self, input):
        batch_mean = input.mean((0, 2, 3), keepdim=True)
        batch_var = input.var((0, 2, 3), keepdim=True)
        batch_std = (batch_var + self.eps).sqrt()
        running_std = (self.running_var + self.eps).sqrt()

        r = torch.clamp(batch_std / running_std, min=1 / self.r_max, max=self.r_max).detach()
        d = torch.clamp((batch_mean - self.running_mean) / running_std, min=-self.d_max, max=self.d_max).detach()

        self.running_mean.lerp_(batch_mean, self.momentum)
        self.running_var.lerp_(batch_var, self.momentum)
        return batch_mean, batch_std, r, d

    def forward(self, input):
        if self.training:
            with torch.no_grad():
                mean, std, r, d = self.update_stats(input)
            input = (input - mean) / std * r + d
        else:
            mean, std = self.running_mean, self.running_var
            input = (input - mean) / (self.running_var + self.eps).sqrt()

        if self.affine:
            return self.weight * input + self.bias
        return input


if __name__ == '__main__':
    m = BatchReNorm2d(100)
    input = torch.randn(20, 100, 35, 45)
    output = m(input)