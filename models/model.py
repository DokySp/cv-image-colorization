from turtle import forward
from numpy import pad
import torch
from torch import nn



class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)



class RecurrentBlock(nn.Module):
    def __init__(self, channels, recurrent_iter=2) -> None:
        super(RecurrentBlock, self).__init__()
        self.recurrent_iter = recurrent_iter
        # self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        if self.recurrent_iter == 0:
            return x
        for i in range(self.recurrent_iter):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1



class R2CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, recurrent_iter=2) -> None:
        super(R2CNNBlock, self).__init__()
        self.rcnn = nn.Sequential(
            RecurrentBlock(out_channels, recurrent_iter),
            RecurrentBlock(out_channels, recurrent_iter),
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.conv(x)
        x1 = self.rcnn(x)
        return x + x1



class AttentionBlock(nn.Module):
    def __init__(self, in_channels, res_channels, psi_channels) -> None:
        super(AttentionBlock, self).__init__()

        # Residual 
        self.W_u = nn.Sequential(
            nn.Conv2d(res_channels, psi_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(psi_channels)
        )

        # Upsampled
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels, psi_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(psi_channels)
        )

        # Linear Transformation
        self.psi = nn.Sequential(
            nn.Conv2d(psi_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, g):
        x1 = self.W_u(x)
        g1 = self.W_g(g)
        
        psi = self.relu(x1+g1)
        psi = self.psi(psi)

        # Mul attention to residual channel
        return x * psi



class AttentionR2Unet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, recurrent_iter=2, start_ch = 64) -> None:
        super(AttentionR2Unet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.r2cnn1 = R2CNNBlock(in_channels, out_channels=start_ch, recurrent_iter=recurrent_iter)
        self.r2cnn2 = R2CNNBlock(start_ch, out_channels=start_ch*2, recurrent_iter=recurrent_iter)
        self.r2cnn3 = R2CNNBlock(start_ch*2, out_channels=start_ch*4, recurrent_iter=recurrent_iter)
        self.r2cnn4 = R2CNNBlock(start_ch*4, out_channels=start_ch*8, recurrent_iter=recurrent_iter)
        self.r2cnn5 = R2CNNBlock(start_ch*8, out_channels=start_ch*16, recurrent_iter=recurrent_iter)

        self.upconv4 = UpConv(start_ch*16, start_ch*8)
        self.attention4 = AttentionBlock(start_ch*8, start_ch*8, start_ch*4)
        self.upr2net4 = R2CNNBlock(start_ch*16, start_ch*8, recurrent_iter=recurrent_iter)

        self.upconv3 = UpConv(start_ch*8, start_ch*4)
        self.attention3 = AttentionBlock(start_ch*4, start_ch*4, start_ch*2)
        self.upr2net3 = R2CNNBlock(start_ch*8, start_ch*4, recurrent_iter=recurrent_iter)
 
        self.upconv2 = UpConv(start_ch*4, start_ch*2)
        self.attention2 = AttentionBlock(start_ch*2, start_ch*2, start_ch*1)
        self.upr2net2 = R2CNNBlock(start_ch*4, start_ch*2, recurrent_iter=recurrent_iter)

        self.upconv1 = UpConv(start_ch*2, start_ch)
        self.attention1 = AttentionBlock(start_ch, start_ch, int(start_ch/2))
        self.upr2net1 = R2CNNBlock(start_ch*2, start_ch, recurrent_iter=recurrent_iter)

        self.conv1 = nn.Conv2d(start_ch, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoding
        x1 = self.r2cnn1(x)

        x2 = self.maxpool(x1)
        x2 = self.r2cnn2(x2)

        x3 = self.maxpool(x2)
        x3 = self.r2cnn3(x3)

        x4 = self.maxpool(x3)
        x4 = self.r2cnn4(x4)

        x5 = self.maxpool(x4)
        x5 = self.r2cnn5(x5)


        # Decoding
        d4 = self.upconv4(x5)
        a4 = self.attention4(x=x4, g=d4)
        d4 = torch.cat((a4, d4), dim=1)
        d4 = self.upr2net4(d4)

        d3 = self.upconv3(d4)
        a3 = self.attention3(x=x3, g=d3)
        d3 = torch.cat((a3, d3), dim=1)
        d3 = self.upr2net3(d3)

        d2 = self.upconv2(d3)
        a2 = self.attention2(x=x2, g=d2)
        d2 = torch.cat((a2, d2), dim=1)
        d2 = self.upr2net2(d2)

        d1 = self.upconv1(d2)
        a1 = self.attention1(x=x1, g=d1)
        d1 = torch.cat((a1, d1), dim=1)
        d1 = self.upr2net1(d1)

        out = self.conv1(d1)

        return out