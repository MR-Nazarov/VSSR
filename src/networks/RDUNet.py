import torch
import torch.nn as nn


@torch.no_grad()
def init_weights(init_type='xavier'):
    if init_type == 'xavier':
        init = nn.init.xavier_normal_
    elif init_type == 'he':
        init = nn.init.kaiming_normal_
    else:
        init = nn.init.orthogonal_

    def initializer(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init(m.weight)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)

    return initializer


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        return self.actv(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, kernel_size, padding=padding)
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.actv = nn.PReLU(out_channels)
        self.actv_t = nn.PReLU(in_channels)

    def forward(self, x):
        upsample, concat = x
        upsample = self.actv_t(self.conv_t(upsample))
        return self.actv(self.conv(torch.cat([concat, upsample], 1)))


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, kernel_size, padding=padding)
        self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, kernel_size, padding=padding)
        self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, kernel_size, padding=padding)
        self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, kernel_size, padding=padding)

        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))
        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))
        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))
        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))
        return out_3 + x


class RDUNet(nn.Module):
    """Residual-Dense U-Net for MRI reconstruction."""

    def __init__(self, in_channels=1, out_channels=1, filters_0=32, kernel_size=(3, 3)):
        super().__init__()

        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0

        kernel_size = tuple(kernel_size)  # accept list from Hydra

        # Encoder
        self.input_block = InputBlock(in_channels, filters_0, kernel_size)
        self.block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0, kernel_size)
        self.block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0, kernel_size)
        self.down_0 = DownsampleBlock(filters_0, filters_1)

        self.block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1, kernel_size)
        self.block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1, kernel_size)
        self.down_1 = DownsampleBlock(filters_1, filters_2)

        self.block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2, kernel_size)
        self.block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2, kernel_size)
        self.down_2 = DownsampleBlock(filters_2, filters_3)

        # Bottleneck
        self.block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3, kernel_size)
        self.block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3, kernel_size)

        # Decoder
        self.up_2 = UpsampleBlock(filters_3, filters_2, filters_2, kernel_size)
        self.block_2_2 = DenoisingBlock(filters_2, filters_2 // 2, filters_2, kernel_size)
        self.block_2_3 = DenoisingBlock(filters_2, filters_2 // 2, filters_2, kernel_size)

        self.up_1 = UpsampleBlock(filters_2, filters_1, filters_1, kernel_size)
        self.block_1_2 = DenoisingBlock(filters_1, filters_1 // 2, filters_1, kernel_size)
        self.block_1_3 = DenoisingBlock(filters_1, filters_1 // 2, filters_1, kernel_size)

        self.up_0 = UpsampleBlock(filters_1, filters_0, filters_0, kernel_size)
        self.block_0_2 = DenoisingBlock(filters_0, filters_0 // 2, filters_0, kernel_size)
        self.block_0_3 = DenoisingBlock(filters_0, filters_0 // 2, filters_0, kernel_size)

        self.output_block = OutputBlock(filters_0, out_channels, kernel_size)

    def forward(self, inputs):
        out_0 = self.input_block(inputs)
        out_0 = self.block_0_0(out_0)
        out_0 = self.block_0_1(out_0)

        out_1 = self.down_0(out_0)
        out_1 = self.block_1_0(out_1)
        out_1 = self.block_1_1(out_1)

        out_2 = self.down_1(out_1)
        out_2 = self.block_2_0(out_2)
        out_2 = self.block_2_1(out_2)

        out_3 = self.down_2(out_2)
        out_3 = self.block_3_0(out_3)
        out_3 = self.block_3_1(out_3)

        out_4 = self.up_2([out_3, out_2])
        out_4 = self.block_2_2(out_4)
        out_4 = self.block_2_3(out_4)

        out_5 = self.up_1([out_4, out_1])
        out_5 = self.block_1_2(out_5)
        out_5 = self.block_1_3(out_5)

        out_6 = self.up_0([out_5, out_0])
        out_6 = self.block_0_2(out_6)
        out_6 = self.block_0_3(out_6)

        # Global residual: add input directly
        return self.output_block(out_6) + inputs[:, 0:1, :, :]
