'''
Building blocks for U-Net written by Julien Martel
Edited by Cindy Nguyen
'''
import torch.nn as nn
import torch


class MiniConvBlock(nn.Module):
    '''
    Implements single conv + ReLU down block
    '''
    def __init__(self, in_ch, out_ch, padding, batch_norm):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=int(padding)))
        blocks.append(nn.ReLU())
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_ch))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding, batch_norm):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=int(padding)))
        blocks.append(nn.ReLU())
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_ch))
        blocks.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=int(padding)))
        blocks.append(nn.ReLU())
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_ch))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode, padding, batch_norm):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            )
        self.conv_block = ConvBlock(in_ch, out_ch, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, img, bridge):
        up = self.up(img)
        crop = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop], 1)

        return self.conv_block(out)


class ConvUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode, padding, batch_norm):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=9, padding=1, groups=in_ch)
            )
        self.conv_block = ConvBlock(in_ch, out_ch, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_x + target_size[0]), diff_x:(diff_x + target_size[1])]

    def upsample(self, layer):
        self.upsample_layer = nn.Upsample(mode='bilinear', scale_factor=2)



    def forward(self, img):
        up = self.up(img)
        return up
        # out = self.conv_block(up)

        # up = self.up(img)
        #
        # print(up.shape)
        # crop = self.center_crop(bridge, up.shape[2:])
        # print(crop.shape)
        # out = torch.cat([up, crop], 1)

        # return self.conv_block(out)