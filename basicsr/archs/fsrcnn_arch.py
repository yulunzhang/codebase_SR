import torch.nn as nn
import math

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, upscale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        
        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )

        # Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
        
        self.mapping = []
        # Non-linear mapping
        for _ in range(m-2):
            self.mapping.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            self.mapping.append(nn.PReLU(s))
            
        self.mapping = nn.Sequential(*self.mapping)
        
        # Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        
        # Deconvolution
        self.deconvolution = nn.ConvTranspose2d(d, 
                                                num_channels, 
                                                kernel_size=9, 
                                                stride=upscale_factor, 
                                                padding=4,
                                                output_padding=upscale_factor-1)
        
    def forward(self, x):
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconvolution(out)
        return out

