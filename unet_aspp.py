import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from utils import *

channels_ = [64,128,256,512]
block_config=(4,8,16,12)
down_times = 3
RGB_input_channels = 3
RGB_output_channel = 1
H_input_channels = 3
H_output_channel = 1
MID_output_channel = 1

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2,dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class down_sample(nn.Module):
    def __init__(self,in_planes, out_planes, stride=2,kernel_size=1, padding=0):
        super(down_sample, self).__init__()
        self.down_sample = nn.Sequential(my_conv(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride,padding=padding),
                        my_conv(out_planes, out_planes, kernel_size=1,
                        stride=1,padding=0))
    def forward(self,x):
        return self.down_sample(x)

class Down(nn.Module): #好像沒有用到這層
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class unet_down_branch(nn.Module):
    def __init__(self,input_channels):
        super(unet_down_branch, self).__init__()
        self.inc = DoubleConv(input_channels, channels_[0])
        #self.down1 = Down(64, 128)
        #self.down2 = Down(128, 256)
        #self.down3 = Down(256, 512)
        self.unetdown1      = unet_block(input_channels, channels_[0])
        self.down_sample1   = down_sample(channels_[0],channels_[0])
        
        self.unetdown2      = unet_block(channels_[0], channels_[1])
        self.down_sample2   = down_sample(channels_[1],channels_[1])
        
        self.unetdown3      = unet_block(channels_[1], channels_[2])
        self.down_sample3   = down_sample(channels_[2],channels_[2])
        
        self.unetdown4      = unet_block(channels_[2], channels_[3])
        
    def forward(self, x):
        ret1 = self.inc(x)
        #print("x_1:",x_1.size())
        #ret2 = self.down1(ret1)
        # print("ret2:",ret2.size())
        #ret3 = self.down2(ret2)
        # print("ret3:",ret3.size())
        #ret4 = self.down3(ret3)
        #print("ret4:",ret4.size())
        #ret5 = self.aspp(ret4)
        #print("ret5:",ret5.size())
        
        #return [ret1,ret2,ret3,ret4,ret5]
        x_1 = self.unetdown1(x)
        #print("x_2:",x_2.size())
        #ret1 = torch.cat((x_1, ret1), dim=1)
        #print("ret1:",ret1.size())
        
        ret2 = self.down_sample1(x_1)
        #print("ret2:",ret2.size())
        ret2 = self.unetdown2(ret2)
        #print("ret2:",ret2.size())
        ret3 = self.down_sample2(ret2)
        ret3 = self.unetdown3(ret3)
        #print("ret3:",ret3.size())
        ret4 = self.down_sample3(ret3) 
        ret4 = self.unetdown4(ret4)
        #print("ret4:",ret4.size())
        return [ret1,ret2,ret3,ret4]
        
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        
class unet_up_sample(nn.Module):
    def __init__(self, inplanes, planes):
        super(unet_up_sample, self).__init__()
        self.up     = up_sample(inplanes, planes)
        self.cat    = my_conv(planes*2, planes,kernel_size=1, padding=0,bias=False)
        self.block  = unet_block(planes,planes)
    def forward(self, up,skip):
        up  = self.up(up)
        cat = self.cat(torch.cat((skip,up),1))
        out = self.block(cat)
        return out
class unet_block(nn.Module):
    def __init__(self,input_size,output_size):
        super(unet_block,self).__init__()
        self.block1 = my_conv(input_size,output_size,kernel_size=3,stride=1, padding=1,bias=False)           
        self.block2 = my_conv(output_size,output_size,kernel_size=3,stride=1, padding=1,bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x
class up_sample(nn.Module):
    def __init__(self,in_planes, out_planes, stride=1,kernel_size=1, padding=0, up_rate=2):
        super(up_sample, self).__init__()
        self.up_rate    = up_rate
        self.out        = my_conv(in_planes, out_planes, stride=1,kernel_size=1, padding=0)
    def forward(self,x):
        return self.out(F.interpolate(x, \
                size=(x.size()[-2]*self.up_rate,x.size()[-1]*self.up_rate),\
                mode='bilinear', align_corners=True))
class my_conv(nn.Module):
    def __init__(self,inplanes, outplanes,kernel_size=3,stride=1,  padding=1,bias=False):
        super(my_conv, self).__init__()
        self.my_conv = nn.Sequential(nn.Conv2d(inplanes, outplanes,kernel_size=kernel_size,\
                        stride=stride,  padding=padding,bias=False),
                        nn.BatchNorm2d(outplanes),
                        nn.ReLU(inplace=True))
    def forward(self,x):
        return self.my_conv(x)                        
class unet_up_branch(nn.Module):
    def __init__(self,output_channel=1, bilinear=True):
        super(unet_up_branch,self).__init__()       
        
        self.bilinear = bilinear

        self.up64   = unet_up_sample(channels_[3],channels_[2])
        self.up128  = unet_up_sample(channels_[2],channels_[1])
        self.up256  = unet_up_sample(channels_[1],channels_[0])

        #self.up1 = Up(1024, 256, bilinear)
        #self.up2 = Up(512, 128, bilinear)
        #self.up3 = Up(256, 64, bilinear)
        #self.up4 = Up(128, 64, bilinear)
        #self.outc = OutConv(64, MID_output_channel)
        
        self.out    = nn.Conv2d(channels_[0],MID_output_channel,kernel_size=1,stride=1, bias=False)
    #def forward(self,down_features):
    #    ret1 = self.up1(down_features[4], down_features[3])
        #print(x.shape)
    #    ret2 = self.up2(ret1, down_features[2])
        #print(x.shape)
    #    ret3 = self.up3(ret2, down_features[1])
        #print(x.shape)
    #    ret4 = self.up4(ret3, down_features[0])
        #print(x.shape)
    #    out = self.outc(ret4)
        #print(x.shape)
    #    out = F.softmax(out,dim=1)

    #    return [ret1,ret2,ret3,ret4],out   
    def forward(self,down_features):
        ret1 = self.up64(down_features[3],down_features[2])
        # print("ret1:",ret1.size())
        ret2 = self.up128(ret1,down_features[1])
        # print("ret2:",ret2.size())
        ret3 = self.up256(ret2,down_features[0])
        # print("ret3:",ret3.size())
        out = torch.sigmoid(self.out(ret3))
        # print("out:",out.size())
        return [ret1,ret2,ret3],out
        
class PDFA(nn.Module):
    def __init__(self,outplanes,fuse_num=4):
    #outplanes denote the output channels of PDFA; fuse_num denote the components of feature to fuse
        super(PDFA, self).__init__()
        self.fuse_num   = fuse_num
        self.pdfa_block = []
        for i in range(self.fuse_num):
            conv = my_conv(outplanes*(i+1), outplanes,kernel_size=5,stride=1, padding=2)
            #conv = DoubleConv(outplanes*(i+1), outplanes)
            self.pdfa_block.append(conv)
        self.pdfa_block = nn.ModuleList(self.pdfa_block)
        
        # self.out = nn.Sequential(my_conv(outplanes,outplanes),
        #                            my_conv(outplanes,outplanes))
        self.out = DoubleConv(outplanes, outplanes)
                                   
    def forward(self,features):
    #the input features,a list,with the same channels
    #example:features = [up_sample_feature or down_sample_feature,RGB,H,skip(only in the upsample)]
    #up_sample_feature or down_sample_feature of the previous block is in the first index of features (features[0])
    #the order of the remaining features is arbitrary , as discussed in the paper in the Table 4
        ret     = None
        input   = features[0]
        temp_features   = []
        for i in range(self.fuse_num):
            #print(input.size())
            ret = self.pdfa_block[i](input)
            temp_features.append(ret) 
            if i >= self.fuse_num-1:
                return self.out(ret)
            
            input = [j.clone() for j in temp_features]
            input.append(features[i+1])
            input = torch.cat(input,1)
        
class segmentation_branch_down(nn.Module):
    def __init__(self):
        super(segmentation_branch_down, self).__init__()
        self.down_conv = []
        for i in range(down_times):
            conv = down_sample(channels_[i], channels_[i+1])
            self.down_conv.append(conv)
        self.down_conv = nn.ModuleList(self.down_conv)
        #self.aspp_mid = _ASPP(512,512)
        
        self.PDFA1 = PDFA(channels_[0],fuse_num=2)#inc
        self.PDFA2 = PDFA(channels_[1],fuse_num=3)
        self.PDFA3  = PDFA(channels_[2],fuse_num=3)
        self.PDFA4  = PDFA(channels_[3],fuse_num=3)
        #self.PDFA5  = PDFA(channels_[3],fuse_num=3)#aspp
    def forward(self,features_h,features_rgb):
    #features_h : is a list of H branch feature with different resolution
    #features_rgb :is a list of RGB branch feature with different resolution
        ret1   = [features_h[0],features_rgb[0]]
        ret1 = self.PDFA1(ret1)
        # print("ret1:",ret1.size())
        ret2   = [self.down_conv[0](ret1),features_h[1],features_rgb[1]]
        # print("self.down_conv[0](ret1):",self.down_conv[0](ret1).size())
        ret2   = self.PDFA2(ret2)
        # print("ret2:",ret2.size())
        ret3   = [self.down_conv[1](ret2),features_h[2],features_rgb[2]]
        #print("ret3:",ret3.size())
        ret3   = self.PDFA3(ret3)
        # print("ret3:",ret3.size())
        ret4   = [self.down_conv[2](ret3),features_h[3],features_rgb[3]]
        #print("ret4:",ret4.size())
        ret4   = self.PDFA4(ret4)
        #print("ret4:",ret4.size())

        #ret5   = [self.aspp_mid(ret4),features_h[4],features_rgb[4]]
        #ret5   = self.PDFA5(ret5)
        #return [ret1,ret2,ret3,ret4,ret5]
        return [ret1,ret2,ret3,ret4]

class segmentation_branch_up(nn.Module):
#segmentation branch for up sample of the Triple U-net
    def __init__(self,conf=None):
        super(segmentation_branch_up, self).__init__()
        self.up_sample = []
        for i in range(down_times):#channels:128to64,256to128,512to256
            conv = my_conv(channels_[i+1],channels_[i])
            self.up_sample.append(conv)
        self.up_sample = nn.ModuleList(self.up_sample)
        
        self.PDFA256 = PDFA(channels_[0],fuse_num=4)
        self.PDFA128 = PDFA(channels_[1],fuse_num=4)
        self.PDFA64  = PDFA(channels_[2],fuse_num=4)

        self.out     = nn.Conv2d(channels_[0],MID_output_channel,kernel_size=1,stride=1, bias=False)
    def forward(self,features_H,features_rgb,mid_features):
        ret    = []

        output = [self.up_sample[2](F.interpolate(mid_features[3], size=features_H[0].size()[2:])),mid_features[2],features_H[0],features_rgb[0]]
        ret.append(self.PDFA64(output))
        
        output = [self.up_sample[1](F.interpolate(ret[0], size=features_H[1].size()[2:])),mid_features[1],features_H[1],features_rgb[1]]
        ret.append(self.PDFA128(output))
        
        output = [self.up_sample[0](F.interpolate(ret[1], size=features_H[2].size()[2:])),mid_features[0],features_H[2],features_rgb[2]]
        output = self.PDFA256(output)

        output = self.out(output)
        output = torch.sigmoid(output)
        
        return output

class net(nn.Module):
    def __init__(self,init=True):
        super(net, self).__init__()
        
        self.H_branch_down      = unet_down_branch(H_input_channels)
        self.H_branch_up        = unet_up_branch(H_output_channel)
        
        self.rgb_branch_down    = unet_down_branch(RGB_input_channels)
        self.rgb_branch_up      = unet_up_branch(RGB_output_channel)

        self.mid_down_block     = segmentation_branch_down()
        self.mid_branch_up      = segmentation_branch_up()

    def forward(self, rgb,H):
        #print("H_branch_down is start!!")
        H_down_features             = self.H_branch_down(H)
        #print(qwrjiow)
        #print("rgb_down_features is start!!")
        rgb_down_features           = self.rgb_branch_down(rgb)
        #print("mid_down_features is start!!")
        mid_down_features           = self.mid_down_block(H_down_features,rgb_down_features)
        #print("H_branch_up is start!!")
        H_up_features,H_output      = self.H_branch_up(H_down_features)
        #print("rgb_branch_up is start!!")
        rgb_up_features,rgb_output  = self.rgb_branch_up(rgb_down_features)
        #print(qwjerio)
        nuclei                      = self.mid_branch_up(H_up_features,rgb_up_features,mid_down_features)
        #print("nuclei is down!!")
        return nuclei,H_output,rgb_output