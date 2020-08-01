import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionNorm2d(nn.Module):
    """Region Normalization with no nn.Parameter
    Parameters
        num_features – C from an expected input of size (N, C, H, W)

    Shape:
        Input: (N, C, H, W) (N, 1, H, W), 1 for foreground regions, 0 for background regions
        Output: (N, C, H, W) (same shape as input)

    Examples:
        >>> m = BasicRegionNorm2d(100)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> mask = torch.zeros(20, 1, 35, 45)
        >>> output = m(input, mask)
    """
    def __init__(self, num_features):
        super(RegionNorm2d, self).__init__()
        self.bn_norm = nn.BatchNorm2d(num_features, affine=False, track_running_stats=False)

    def forward(self, x, label):
        label = label.detach()
        rn_foreground_region = self.region_norm(x * label, label)
        rn_background_region = self.region_norm(x * (1 - label), 1 - label)

        return rn_foreground_region + rn_background_region

    def region_norm(self, region, mask):
        shape = region.size()

        sum = torch.sum(region, dim=[0, 2, 3])  # (B, C) -> (C)
        Sr = torch.sum(mask, dim=[0, 2, 3])  # (B, 1) -> (1)
        Sr[Sr == 0] = 1
        mu = (sum / Sr)  # (B, C) -> (C)

        return self.bn_norm(region + (1 - mask) * mu[None,:,None,None]) * \
        (torch.sqrt(Sr / (shape[0] * shape[2] * shape[3])))[None,:,None,None]


class BasicRegionNorm2d(nn.Module):
    def __init__(self, num_features):
        super(BasicRegionNorm2d, self).__init__()
        """Basic Region Normalization
        Parameters
            num_features – C from an expected input of size (N, C, H, W)

        Shape:
            Input: (N, C, H, W) (N, 1, H, W), 1 for foreground regions, 0 for background regions
            Output: (N, C, H, W) (same shape as input)

        Examples:
            >>> m = BasicRegionNorm2d(100)
            >>> input = torch.randn(20, 100, 35, 45)
            >>> mask = torch.zeros(20, 1, 35, 45)
            >>> output = m(input, mask)
        """
        self.rn = RegionNorm2d(num_features)

        self.foreground_gamma = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(num_features), requires_grad=True)

    def forward(self, x, mask):
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')
        rn_x = self.rn(x, mask)

        rn_x_foreground = (rn_x * mask) * (
            1 + self.foreground_gamma[None, :, None, None]) + self.foreground_beta[None, :, None, None]
        rn_x_background = (rn_x * (1 - mask)) * (
            1 + self.background_gamma[None, :, None, None]) + self.background_beta[None, :, None, None]

        return rn_x_foreground + rn_x_background


class SelfAwareAffine(nn.Module):
    def __init__(self, kernel_size=7):
        super(SelfAwareAffine, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.gamma_conv = nn.Conv2d(1, 1, kernel_size, padding=padding)
        self.beta_conv = nn.Conv2d(1, 1, kernel_size, padding=padding)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        x = self.conv1(x)
        importance_map = self.sigmoid(x)

        gamma = self.gamma_conv(importance_map)
        beta = self.beta_conv(importance_map)

        return importance_map, gamma, beta


class LearnableRegionNorm2d(nn.Module):
    def __init__(self, feature_channels, threshold=0.8):
        super(LearnableRegionNorm2d, self).__init__()
        """Basic Region Normalization
        Parameters
            num_features – C from an expected input of size (N, C, H, W)

        Shape:
            Input: (N, C, H, W)
            Output: (N, C, H, W) (same shape as input)

        Examples:
            >>> m = LearnableRegionNorm2d(100)
            >>> input = torch.randn(20, 100, 35, 45)
            >>> output = m(input)
        """
        self.threshold = threshold
        self.sa = SelfAwareAffine()
        self.rn = RegionNorm2d(feature_channels)

    def forward(self, x):
        sa_map, gamma, beta = self.sa(x)
        mask = torch.zeros_like(sa_map).to(x.device)
        mask[sa_map.detach() >= self.threshold] = 1

        rn_x = self.rn(x, mask.expand(x.size()))

        rn_x = rn_x * (1 + gamma) + beta

        return rn_x


if __name__ == '__main__':
    m = BasicRegionNorm2d(100)
    input = torch.randn(20, 100, 35, 45)
    mask = torch.zeros(20, 1, 35, 45)
    output = m(input, mask)

    m = LearnableRegionNorm2d(100)
    input = torch.randn(20, 100, 35, 45)
    output = m(input)