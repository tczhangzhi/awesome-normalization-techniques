import torch
import torch.nn as nn
import torch.nn.functional as F


class TLU(nn.Module):
    def __init__(self, num_features):
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def forward(self, x):
        return torch.max(x, self.tau)


class FilterResponseNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_eps_leanable=False):
        """PyTorch-Filter Response Normalization
        SHOULD be utilized with TLU, highly recommend referring to the OFFICIAL project.

        Parameters
            num_features – C from an expected input of size (N, C, H, W)
            eps – a value added to the denominator for numerical stability. Default: 1e-6

        Shape:
            Input: (N, C, H, W)
            Output: (N, C, H, W) (same shape as input)

        Examples:
            >>> m = FilterResponseNorm2d(100)
            >>> input = torch.randn(20, 100, 35, 45)
            >>> output = m(input)
        """
        super(FilterResponseNorm2d, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable

        self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        x = self.weight * x + self.bias
        return x


if __name__ == '__main__':
    m = FilterResponseNorm2d(100)
    input = torch.randn(20, 100, 35, 45)
    output = m(input)