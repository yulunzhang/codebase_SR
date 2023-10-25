import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
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
        out1 = self.mapping(out)
        out = self.expanding(out1)
        out = self.deconvolution(out)
        return out

if __name__ == "__main__":
    upscale = 3
    height = 80
    width = 71
    model = FSRCNN(upscale_factor=upscale, num_channels=3, m=2)
    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)

    