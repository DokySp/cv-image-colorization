from tokenize import Double
import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1) -> None:
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*2, out_channels, kernel_size, stride, padding)
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride)
        )

        self.sig = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out) * x2
        out = self.sig(out)

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1) -> None:
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  
        )

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.conv(x) + self.res(x)

img_size = 128

class AttentionR2Unet(nn.Module):

    def __init__(self) -> None:
        super(AttentionR2Unet, self).__init__()

        # Encode
        self.double_conv1 = DoubleConv(3, img_size, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2)

        self.double_conv2 = DoubleConv(img_size, img_size*2, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2)

        self.double_conv3 = DoubleConv(img_size*2, img_size*4, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2)

        self.double_conv4 = DoubleConv(img_size*4, img_size*8, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2)

        # Bridge
        self.double_conv7 = DoubleConv(img_size*8, img_size*16, 3, 1, 1)

        # ?
        self.double_conv5 = DoubleConv(img_size*32, img_size*16, 3, 1, 1)


        # Decode
        
    def forward(self, inputs):

        # Encode
        conv1 = self.double_conv1(inputs)
        out = self.maxpool1(conv1)

        conv2 = self.double_conv2(out)
        out = self.maxpool2(conv2)

        conv3 = self.double_conv3(out)
        out = self.maxpool3(conv3)

        conv4 = self.double_conv4(out)
        out = self.maxpool4(out)

        out = self.double_conv5(out)

