import torch.nn as nn
from typing import Any
from .swin_transformer_3d import SwinTransformer3D
from . import unet_decoder

class OmniUnet(nn.Module):
    def __init__(self, max_channels, n_classes, bilinear = False, depth_mode = "summed_tokens", pretrained = False, frozen_stages = -1):
        super(OmniUnet, self).__init__()
        self.n_classes  = n_classes
        self.bilinear   = bilinear
        self.depth_mode = depth_mode

        factor = 2 if bilinear else 1
        
        self.up1  = unet_decoder.Up(1024, 512 // factor, bilinear)
        self.up2  = unet_decoder.Up(512, 256  // factor, bilinear)
        self.up3  = unet_decoder.Up(256, 128  // factor, bilinear)
        self.up4  = unet_decoder.Up(128, 64, bilinear)
        self.outc = unet_decoder.OutConv(64, n_classes)

        self.trunk = omnivore_swinB_UNET(max_channels = max_channels, depth_mode = depth_mode, pretrained = pretrained, frozen_stages = -1)


    def forward(self,x, image_type = ""):
        omnivore_features = self.trunk(x, image_type = image_type)

        x1 = omnivore_features[0][0]
        x1 = x1.squeeze(2)
        x2 = omnivore_features[0][1]
        x2 = x2.squeeze(2)
        x3 = omnivore_features[0][2]
        x3 = x3.squeeze(2)
        x4 = omnivore_features[0][3]
        x4 = x4.squeeze(2)
        x5 = omnivore_features[0][4]
        x5 = x5.squeeze(2)

        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        logits = self.outc(x)

        omnivore_ffeat = omnivore_features[0][5].cuda()
        head           = nn.Linear(in_features = omnivore_ffeat.size()[1], out_features = self.n_classes, bias = True).cuda()
        omnivore_only  = head(omnivore_ffeat).cuda()
        
        return logits, omnivore_only


def omnivore_swinB_UNET(
    pretrained: bool = False,
    progress: bool = True,
    load_heads: bool = True,
    checkpoint_name: str = "omnivore_swinB",
    depth_mode = "summed_tokens",
    max_channels = 3,
    frozen_stages = -1,
    **kwargs: Any,
) -> nn.Module:
    
    trunk = SwinTransformer3D(
        pretrained=pretrained,
        patch_size=(2, 4, 4),
        embed_dim=64,
        depths=[2, 2, 2, 18, 2],
        num_heads=[4, 8, 16, 32, 64],
        window_size=(16, 7, 7),
        drop_path_rate=0.3,
        patch_norm=True,
        frozen_stages = frozen_stages,
        depth_mode = depth_mode,
        max_channels = max_channels,
        **kwargs,
    )

    return trunk

