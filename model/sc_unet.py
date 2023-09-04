
from model.sc_unet import *

from torch import nn

class EncoderDecoder(nn.Module):

    def __init__(self, dim=48, mode=None, mode2=None, residual=False, resblock=False):
        super(EncoderDecoder, self).__init__()
        self.mode = mode
        self.mode2 = mode2

        if self.mode == "feature6" :
            self.encoder = SC_UNET_Encoder(dim=dim, mode=self.mode)
            if self.mode2 is None:
                self.decoder = SC_UNET_Decoder(dim=dim*32, mode=self.mode)
            else:
                if residual:
                    self.decoder = SC_UNET_res_Decoder(dim=dim*32, mode=self.mode, mode2=self.mode2)
                elif resblock:
                    self.decoder = SC_UNET_ResBlock_Decoder(dim=dim*32, mode=self.mode, mode2=self.mode2)
                else:
                    self.decoder = SC_UNET_Decoder(dim=dim*32, mode=self.mode, mode2=self.mode2)
                self.conv = nn.Conv2d(dim*2, 3, 1, 1)
                self.sigmoid = nn.Sigmoid()

        else:
            self.encoder = SC_UNET_Encoder(dim=dim)
            self.decoder = SC_UNET_Decoder(dim=dim*16)


    def forward(self, x):
        if self.mode == 'level5' :
            f5, f4, f3, f2, f1 = self.encoder(x)
            d5, d4, d3, d2, d1, out = self.decoder([f5, f4, f3, f1])

        elif self.mode == 'level4':
            f5, f4, f3, f2, f1 = self.encoder(x)
            d5, d4, d3, d2, d1, out = self.decoder([f5, f4, f3, f2])

        elif self.mode == 'feature6':
            f6, f5, f4, f3, f2, f1 = self.encoder(x)

            if self.mode2 == 'level123':
                d6, d5, d4, d3, d2, d1, out = self.decoder([f6, f5, f4, f3])
                
                out_half = self.conv(d2)
                out_half = self.sigmoid(out_half)

                return out, out_half, [f6, f5, f4, f3, f2, f1], [d6, d5, d4, d3, d2, d1]
 
            else:
                d6, d5, d4, d3, d2, d1, out = self.decoder([f6, f5, f4])
            return out, [f6, f5, f4, f3, f2, f1], [d6, d5, d4, d3, d2, d1]
        
        
        else: # original model
            f5, f4, f3, f2, f1 = self.encoder(x)
            d5, d4, d3, d2, d1, out = self.decoder([f5, f4, f3])
        

        return out, [f5, f4, f3, f2, f1], [d5, d4, d3, d2, d1]
    
