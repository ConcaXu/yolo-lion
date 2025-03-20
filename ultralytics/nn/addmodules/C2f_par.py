import torch
import torch.nn as nn



class Conv(nn.Module):
    """Standard convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ParallelConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ParallelConvolution, self).__init__()
        self.conv1x1 = Conv(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = Conv(in_channels, out_channels, kernel_size=3)
        self.conv5x5 = Conv(in_channels, out_channels, kernel_size=5)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3(x)
        out3 = self.conv5x5(x)

        # Sum the outputs along the channel dimension
        out = (out1 + out2 + out3) / 3  # Average the outputs
        return out


class Bottleneck_Parallel(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, and expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ParallelConvolution(c1, c_)
        self.cv2 = Conv(c_, c2, kernel_size=3, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_Parallel(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_Parallel(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
