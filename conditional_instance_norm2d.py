import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalInstanceNorm2d(nn.Module):
    """Conditional Instance Normalization
    Parameters
        num_features – C from an expected input of size (N, C, H, W)
        num_classes – Number of classes in the datset.
        bias – if set to True, adds a bias term to the embedding layer

    Shape:
        Input: (N, C, H, W) (N,)
        Output: (N, C, H, W) (same shape as input)

    Examples:
        >>> m = ConditionalInstanceNorm2d(100, 10)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> label = torch.randint(10, (20,))
        >>> output = m(input, label)
    """
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        h = self.instance_norm(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


if __name__ == '__main__':
    m = ConditionalInstanceNorm2d(100, 10)
    input = torch.randn(20, 100, 35, 45)
    label = torch.randint(10, (20, ))
    output = m(input, label)