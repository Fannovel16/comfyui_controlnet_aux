#https://github.com/siyeong0/Anime-Face-Segmentation/blob/main/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import MobileNet_V2_Weights

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.NUM_SEG_CLASSES = 7 # Background, hair, face, eye, mouth, skin, clothes
        
        mobilenet_v2 = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        mob_blocks = mobilenet_v2.features
        
        # Encoder
        self.en_block0 = nn.Sequential(    # in_ch=3 out_ch=16
            mob_blocks[0],
            mob_blocks[1]
        )
        self.en_block1 = nn.Sequential(    # in_ch=16 out_ch=24
            mob_blocks[2],
            mob_blocks[3],
        )
        self.en_block2 = nn.Sequential(    # in_ch=24 out_ch=32
            mob_blocks[4],
            mob_blocks[5],
            mob_blocks[6],
        )
        self.en_block3 = nn.Sequential(    # in_ch=32 out_ch=96
            mob_blocks[7],
            mob_blocks[8],
            mob_blocks[9],
            mob_blocks[10],
            mob_blocks[11],
            mob_blocks[12],
            mob_blocks[13],
        )
        self.en_block4 = nn.Sequential(    # in_ch=96 out_ch=160
            mob_blocks[14],
            mob_blocks[15],
            mob_blocks[16],
        )
        
        # Decoder
        self.de_block4 = nn.Sequential(     # in_ch=160 out_ch=96
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(160, 96, kernel_size=3, padding=1),
            nn.InstanceNorm2d(96),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2)
        )
        self.de_block3 = nn.Sequential(     # in_ch=96x2 out_ch=32
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(96*2, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2)
        )
        self.de_block2 = nn.Sequential(     # in_ch=32x2 out_ch=24
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32*2, 24, kernel_size=3, padding=1),
            nn.InstanceNorm2d(24),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2)
        )
        self.de_block1 = nn.Sequential(     # in_ch=24x2 out_ch=16
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(24*2, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2)
        )
        
        self.de_block0 = nn.Sequential(     # in_ch=16x2 out_ch=7
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16*2, self.NUM_SEG_CLASSES, kernel_size=3, padding=1),
            nn.Softmax2d()
        )
        
    def forward(self, x):
        e0 = self.en_block0(x)
        e1 = self.en_block1(e0)
        e2 = self.en_block2(e1)
        e3 = self.en_block3(e2)
        e4 = self.en_block4(e3)
        
        d4 = self.de_block4(e4)
        c4 = torch.cat((d4,e3),1)
        d3 = self.de_block3(c4)
        c3 = torch.cat((d3,e2),1)
        d2 = self.de_block2(c3)
        c2 =torch.cat((d2,e1),1)
        d1 = self.de_block1(c2)
        c1 = torch.cat((d1,e0),1)
        y = self.de_block0(c1)
        
        return y