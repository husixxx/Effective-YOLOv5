# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""
import warnings
import torch
import torch.nn as nn
from models.common import Conv


class BottleneckPruned(nn.Module):
    # Pruned bottleneck
    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1):  # ch_in, ch_out, shortcut, groups, expansion
        super(BottleneckPruned, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out, cv2out, 3, 1, g=g)
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3Pruned(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, cv1in, cv1out, cv2out, cv3out, bottle_args, n=1, shortcut=True, g=1):
        super(C3Pruned, self).__init__()
        cv3in = bottle_args[-1][-1]
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)
        self.cv3 = Conv(cv3in + cv2out, cv3out, 1)
        self.m = nn.Sequential(*[BottleneckPruned(*bottle_args[k], shortcut, g) for k in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPPFPruned(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, cv1in, cv1out, cv2out, k=5):
        super(SPPFPruned, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out * 4, cv2out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
    # Handle mismatch
        if x.shape[1] != self.cv1.conv.weight.shape[1]:
            print(f"SPPFPruned mismatch: {x.shape[1]} vs {self.cv1.conv.weight.shape[1]}")
            if x.shape[1] < self.cv1.conv.weight.shape[1]:
                padding = torch.zeros(x.shape[0],
                                    self.cv1.conv.weight.shape[1] - x.shape[1],
                                    x.shape[2],
                                    x.shape[3],
                                    device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :self.cv1.conv.weight.shape[1], :, :]
                
        # Rest of existing code
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
