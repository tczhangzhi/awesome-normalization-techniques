import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveNorm2d(nn.BatchNorm2d):
    """Batch Re-Normalization
    Parameters
        num_features – C from an expected input of size (N, C, H, W)
        eps – a value added to the denominator for numerical stability. Default: 1e-5
        momentum – the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
        hidden_channels – the size of the hidden layer. Default: 32

    Shape:
        Input: (N, C, H, W)
        Output: (N, C, H, W) (same shape as input)

    Examples:
        >>> m = AttentiveNorm2d(100)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """
    def __init__(self, num_features, hidden_channels=32, eps=1e-5, momentum=0.1, track_running_stats=True):
        super(AttentiveNorm2d, self).__init__(num_features,
                                              eps=eps,
                                              momentum=momentum,
                                              affine=False,
                                              track_running_stats=track_running_stats)

        self.gamma = nn.Parameter(torch.Tensor(hidden_channels, num_features))
        self.beta = nn.Parameter(torch.Tensor(hidden_channels, num_features))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, hidden_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = super(AttentiveNorm2d, self).forward(x)

        size = output.size()
        b, c, _, _ = x.size()

        y = self.avgpool(x).view(b, c)
        y = self.fc(y)
        y = self.sigmoid(y)

        gamma = y @ self.gamma
        beta = y @ self.beta

        gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand(size)
        beta = beta.unsqueeze(-1).unsqueeze(-1).expand(size)

        return gamma * output + beta


if __name__ == '__main__':
    m = AttentiveNorm2d(100)
    input = torch.randn(20, 100, 35, 45)
    output = m(input)