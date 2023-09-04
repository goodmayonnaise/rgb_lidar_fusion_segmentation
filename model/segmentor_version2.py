
import torch 
from torch import nn

from model.adapter import SalsaNextAdapter
from model.salsanext import SalsaNextEncoder, SalsaNextDecoder
from model.modules.adapter_modules import deform_inputs
from model.adapter import PatchEmbed

import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    def __init__(self, nclasses):
        super(EncoderDecoder, self).__init__()
        self.img_size = (256, 1024)
        self.adapter = SalsaNextAdapter(img_size=self.img_size)
        self.backbone = SalsaNextEncoder(nclasses)
        self.embed_dim = 768
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.patch_embed1 = PatchEmbed(img_size=self.img_size, patch_size=16, in_chans=64, embed_dim=768)
        self.patch_embed2 = PatchEmbed(img_size=self.img_size, patch_size=16, in_chans=32, embed_dim=768)
      
        self.conv1x1_64 = nn.Conv2d(64, 32, 1, 1)
        self.conv1x1_128 = nn.Conv2d(128, 32, 1, 1)
        self.conv1x1_256 = nn.Conv2d(256, 32, 1, 1)
        self.conv1x1_384 = nn.Conv2d(768, 384, 1, 1)
        self.conv1x1_192 = nn.Conv2d(768, 192, 1, 1)

        self.decode_head = SalsaNextDecoder(nclasses)

    def intercationblock(self, x, c, cls, deform_inputs1, deform_inputs2, bs, dim, H, W, idx=0):
        indexes = self.adapter.interaction_indexes[idx]
        layer = self.adapter.interactions[idx]
        x, c, cls = layer(x, c, cls, self.adapter.blocks[indexes[0]:indexes[-1]+1], deform_inputs1, deform_inputs2, H, W)
        
        return x, c, cls

    def forward(self, img, rdm):

        # adapter spm 
        c1, c2, c3, c4 = self.adapter.spm(rdm) # 768 64 256 | 4096 768 | 1024 768 | 256 768
        c2, c3, c4 = self.adapter._add_level_embed(c2, c3, c4) # 4096 768 | 1024 768 | 256 768 
        c = torch.cat([c2, c3, c4], dim=1)  # 5376 768

        outs = list()
        # salsanext context module
        downCntx = self.backbone.downCntx(img)
        downCntx = self.backbone.downCntx2(downCntx)
        downCntx = self.backbone.downCntx3(downCntx) # 1 32 1024 256

        # salsanext block1
        down0c, down0b = self.adapter.resBlock1(downCntx) # 64 512 128
        down0cc = F.interpolate(down0c, scale_factor=2, mode='bilinear', align_corners=True) # 64 1024 256
        # deform prepare1
        deform_inputs1, deform_inputs2 = deform_inputs(down0cc) # 1 1024 1 2 | 3 2 | 3            | 1 5376 1 2 | 1 2 | 1
        # patch 1
        x, H, W = self.patch_embed1(down0cc) # 1024 768 | 16 | 64
        bs, n, dim = x.shape 
        cls = self.cls_token.expand(bs, -1, -1) # 1 1 768
        # adapter1
        x, c, cls = self.intercationblock(x, c, cls, deform_inputs1, deform_inputs2, bs, dim, H, W, idx=0) # 1024 768 | 5376 768 | 1
        outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous()) # 768 64 16

        # salsanext block2
        down1c, down1b = self.adapter.resBlock2(down0c) 
        down1cc = self.conv1x1_128(down1c)
        down1cc = F.interpolate(down1cc, scale_factor=4, mode='bilinear', align_corners=True)
        deform_inputs1, deform_inputs2 = deform_inputs(down1cc)
        cls = self.cls_token.expand(bs, -1, -1)        
        x, H, W = self.patch_embed2(down1cc) # 64 16
        bs, n, dim = x.shape
        x, c, cls = self.intercationblock(x, c, cls, deform_inputs1, deform_inputs2, bs, dim, H, W, idx=1) # 1024 768 | 5376 768 | 1 768
        outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous()) # 768 64 16

        # salsanext block3
        down2c, down2b = self.adapter.resBlock3(down1c) 
        down2cc = self.conv1x1_256(down2c)
        down2cc = F.interpolate(down2cc, scale_factor=8, mode='bilinear', align_corners=True)
        deform_inputs1, deform_inputs2 = deform_inputs(down2cc)
        cls = self.cls_token.expand(bs, -1, -1)        
        x, H, W = self.patch_embed2(down2cc) # 64 16
        bs, n, dim = x.shape
        x, c, cls = self.intercationblock(x, c, cls, deform_inputs1, deform_inputs2, bs, dim, H, W, idx=2) # 1024 768 | 5376 768 | 1 768
        outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # salsanext block4
        down3c, down3b = self.adapter.resBlock4(down2c) # 256 64 16
        down3cc = self.conv1x1_256(down3c)
        down3cc = F.interpolate(down3cc, scale_factor=16, mode='bilinear', align_corners=True)
        deform_inputs1, deform_inputs2 = deform_inputs(down3cc)
        cls = self.cls_token.expand(bs, -1, -1)
        x, H, W = self.patch_embed2(down3cc)
        bs, n, dim = x.shape
        x, c, cls = self.intercationblock(x, c, cls, deform_inputs1, deform_inputs2, bs, dim, H, W, idx=3)
        outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous()) # 768 64 16

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.adapter.up(c2) + c1      

        # add 
        x1, x2, x3, x4 = outs
        x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=True)
        c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        f1 = self.adapter.norm1(c1)
        f2 = self.adapter.norm2(c2)
        f3 = self.adapter.norm3(c3)
        f4 = self.adapter.norm4(c4)

        # ENCODER END - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        # DECODER START  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        up4e = self.decode_head.upBlock1(f4, f3) # 384 64 16
        f2 = self.conv1x1_384(f2) # 384 128 32
        up3e = self.decode_head.upBlock2(up4e, f2) # 192 128 32
        f1 = self.conv1x1_192(f1)
        up2e = self.decode_head.upBlock3(up3e, f1) # 96 256 64
        up1e = F.interpolate(up2e, scale_factor=4, mode='bilinear', align_corners=False) # c = 96

        logits = self.decode_head.logits(up1e) # 20 256 1024 # input c 96
        logits = F.softmax(logits, dim=1) # 20 256 1024

        return logits
