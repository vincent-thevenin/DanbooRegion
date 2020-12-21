import torch
import torch.nn as nn

class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1, stride=1):
        super(ResBlockDown, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.relu = nn.LeakyReLU()
        self.relu_inplace = nn.LeakyReLU(inplace = True)
        #self.avg_pool2d = nn.AvgPool2d(2)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, stride=stride))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size, stride=stride))

    def forward(self, x):
        res = x
        
        #left
        if self.in_channel != self.out_channel:
            out_res = self.conv_l1(res)
            #out_res = self.avg_pool2d(out_res)
        else:
            out_res = res
        
        #right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        #out = self.avg_pool2d(out)
        
        #merge
        out = out_res + out
        
        return out

class ResBlockUp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size=None, scale = 2, conv_size=3, padding_size = 1, is_bilinear = True):
        super(ResBlockUp, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        if is_bilinear:
            self.upsample = nn.Upsample(size = out_size, scale_factor=scale, mode='bilinear')
        else:
            self.upsample = nn.Upsample(size = out_size, scale_factor=scale)
        self.relu = nn.LeakyReLU(inplace = True)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1))
        
        #right
        self.norm1 = nn.InstanceNorm2d(in_channel, affine=True)
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
        self.norm2 = nn.InstanceNorm2d(out_channel, affine=True)
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))
    
    def forward(self,x):        
        res = x
        
        #left
        if self.in_channel != self.out_channel:
            out_res = self.upsample(res)
            out_res = self.conv_l1(out_res)
        else:
            out_res = self.upsample(res)
            out_res = res
        
        #right
        out = self.norm1(x)
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv_r2(out)
        
        out = out + out_res
        
        return out

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

        self.model = [
            ResBlockDown(3, 64, stride=2)
        ]

        #down
        for _ in range(4):
            self.model.append(ResBlockDown(64,64))
        
        self.model.append(ResBlockDown(64,128,stride=2))
        for _ in range(4):
            self.model.append(ResBlockDown(128,128))

        self.model.append(ResBlockDown(128,256,stride=2))
        for _ in range(4):
            self.model.append(ResBlockDown(256,256))

        #up
        self.model.append(ResBlockUp(256,128, scale=2))
        for _ in range(4):
            self.model.append(ResBlockUp(128,128, scale=1))

        self.model.append(ResBlockUp(128,64, scale=2))
        for _ in range(4):
            self.model.append(ResBlockUp(64,64, scale=1))

        self.model.append(ResBlockUp(64,32, scale=2))
        self.model.append(nn.Conv2d(32,16,3,padding=1))
        self.model.append(nn.LeakyReLU(inplace=True))
        self.model.append(nn.Conv2d(16,8,3,padding=1))
        self.model.append(nn.LeakyReLU(inplace=True))
        self.model.append(nn.Conv2d(8,1,3,padding=1))
        self.model.append(nn.Sigmoid())

        self.model = nn.Sequential(*self.model)

    def forward(self,x):
        return self.model(x)