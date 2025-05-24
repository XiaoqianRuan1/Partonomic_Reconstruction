from copy import deepcopy
import torch
from torch import nn
from torchvision import models as tv_models

from .tools import get_output_size, Identity, kaiming_weights_init
from .unet import UNet
from .pos_encoding import build_position_encoding
import numpy as np
from typing import List
from torch import Tensor
from .pos_encoding import NestedTensor
import torchvision

from copy import deepcopy
import torch
from torch import nn
from torchvision import models as tv_models
from .resnet_dilated import ResnetDilated
from .tools import get_output_size, Identity, kaiming_weights_init
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict

def get_resnet_model(name):
    if name is None:
        name = 'resnet18'
    return {
        'resnet18': tv_models.resnet18,
        'resnet34': tv_models.resnet34,
        'resnet50': tv_models.resnet50,
        'resnet101': tv_models.resnet101,
        'resnet152': tv_models.resnet152,
        'resnext50_32x4d': tv_models.resnext50_32x4d,
        'resnext101_32x8d': tv_models.resnext101_32x8d,
        'wide_resnet50_2': tv_models.wide_resnet50_2,
        'wide_resnet101_2': tv_models.wide_resnet101_2,
    }[name]

"""
class Encoder(nn.Module):
    def __init__(self, out_channels=128, dims=16, nres_block=2, normalizer_fn=None, demosaic=False, use_center=False,
                 use_noise_map=False):
        super(Encoder, self).__init__()

        self.out_channels = out_channels
        self.dims = dims
        self.nres_block = nres_block
        self.normalizer_fn = normalizer_fn
        self.demosaic = demosaic
        self.use_center = use_center
        self.use_noise_map = use_noise_map
        self.out_ch=128

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=dims, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=dims, out_channels=dims * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=dims * 2, out_channels=dims * 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=dims * 4, out_channels=dims * 8, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=dims * 8, out_channels=dims * 16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=dims * 16, out_channels=dims * 12, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=dims * 12, out_channels=dims * 10, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=dims * 10, out_channels=dims*9, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=dims * 9, out_channels=out_channels, kernel_size=1, padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, inputs):
        #print("input image shape is ", inputs.shape)

        conv1 = F.leaky_relu(self.conv1(inputs))
        conv2 = F.leaky_relu(self.conv2(conv1))
        pool1 = F.max_pool2d(conv2, kernel_size=2)
        conv3 = F.leaky_relu(self.conv3(pool1))
        pool2 = F.max_pool2d(conv3, kernel_size=2)
        conv4 = F.leaky_relu(self.conv4(pool2))
        pool3 = F.max_pool2d(conv4, kernel_size=2)
        conv5 = F.leaky_relu(self.conv5(pool3))
        conv6 = F.leaky_relu(self.conv6(conv5))

        up7 = F.interpolate(conv6, scale_factor=2, mode="nearest")
        up7 = self.conv7(up7)
        up7 = torch.cat([up7, conv4], dim=1)
        conv7 = F.leaky_relu(self.conv7(up7))

        up8 = F.interpolate(conv7, scale_factor=2, mode="nearest")
        up8 = self.conv8(up8)
        up8 = torch.cat([up8, conv3], dim=1)
        conv8 = F.leaky_relu(self.conv8(up8))

        up9 = F.interpolate(conv8, scale_factor=2, mode="nearest")
        up9 = self.conv9(up9)
        up9 = torch.cat([up9, conv2], dim=1)
        conv9 = F.leaky_relu(self.conv9(up9))
        
        conv10 = self.conv10(conv9)
        out = conv10
        return out
"""

class GlobalEncoder(nn.Module):
    color_channels = 3

    def __init__(self, img_size, name='resnet18', **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.with_pool = kwargs.pop('with_pool', True)
        pretrained = kwargs.pop('pretrained', False)
        n_features = kwargs.pop('n_features', 128)
        assert len(kwargs) == 0
        if name == 'identity':
            self.encoder = Identity()
        else:
            resnet = get_resnet_model(name)(pretrained=pretrained, progress=False)
            seq = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                   resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
            if self.with_pool:
                size = self.with_pool if isinstance(self.with_pool, (tuple, list)) else (1, 1)
                seq.append(torch.nn.AdaptiveAvgPool2d(output_size=size))
            self.encoder = nn.Sequential(*seq)

        out_ch = get_output_size(self.color_channels, img_size, self.encoder)
        fc = nn.Sequential()
        if n_features is not None:
            if out_ch != n_features:
                assert n_features < out_ch
                fc = nn.Linear(out_ch, n_features)
                _ = kaiming_weights_init(fc)
                out_ch = n_features
        self.out_ch = out_ch
        self.fc = fc

    def forward(self, x):
        return self.fc(self.encoder(x).flatten(1))

class AttentionBlock(nn.Module):
    """
    residual attention module
    """
    color_channels = 3
    
    def __init__(self, img_size, name='resnet18', **kwargs):
        super().__init__()
        self.with_pool = kwargs.pop('with_pool', True)
        pretrained = kwargs.pop('pretrained', False)
        resnet = get_resnet_model(name)(pretrained=pretrained, progress=False)
        seq = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool] # 32x32x64
        self.encoder = nn.Sequential(*seq)
        # attention module
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.upsample1 = nn.UpsamplingBilinear2d(size=16)
        self.upsample2 = nn.UpsamplingBilinear2d(size=8)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        #self.bn3 = nn.BatchNorm2d(256)
        self.sigmoid = nn.Sigmoid()
        self.upsample3 = nn.UpsamplingBilinear2d(size=4)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        #self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self,x):
        output = self.encoder(x) # 16x16x64
        conv2 = self.conv2(output) # 16x16x128
        output = self.bn(conv2)
        #output = self.relu(output)
        mask1 = self.sigmoid(conv2) # 32x32x128
        output = self.maxpool(output) # 8x8x128
        
        conv3 = self.conv3(output) # 8x8x128
        mask2 = self.sigmoid(conv3)
        output = self.bn(conv3)
        #output = self.relu(output)
        output = self.maxpool(output) # 4x4x128
        
        conv4 = self.conv4(output) # 4x4x128
        mask3 = self.sigmoid(conv4)
        output = self.bn(conv4)
        #output = self.relu(output)
        output = self.maxpool(output)# 4x4x128
        
        result = self.upsample3(output)
        sum = mask3 * result + result
        sum = self.upsample2(sum)
        result = mask2 * sum + sum
        result = self.upsample1(result)
        sum = mask1 * result + result
        return sum

class ResidualAttention(nn.Module):
    color_channels = 3
    
    def __init__(self, img_size, name='resnet18',**kwargs):
        super().__init__()
        n_features = kwargs.pop('n_features')
        self.att_encoder = AttentionBlock(img_size, **kwargs)
        out_ch = get_output_size(self.color_channels,img_size,self.att_encoder)
        #size = self.with_pool if isinstance(self.with_pool, (tuple, list)) else (1, 1)
        #self.pool = torch.nn.AdaptiveAvgPool2d(output_size=size)
        fc = nn.Sequential()
        if n_features is not None:
            if out_ch != n_features:
                assert n_features < out_ch
                fc = nn.Linear(out_ch, n_features)
                _ = kaiming_weights_init(fc)
                out_ch = n_features # 512
        self.out_ch = out_ch
        self.fc = fc
        
    def forward(self,x):
        output = self.att_encoder(x)
        #result = self.pool(output)
        return self.fc(output.flatten(1))

class PartEncoder(nn.Module):
    color_channels = 3
    
    def __init__(self, img_size, name='resnet18', **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.with_pool = kwargs.pop('with_pool', True)
        pretrained = kwargs.pop('pretrained', False)
        assert len(kwargs) == 0
        if name == 'identity':
            self.encoder = Identity()
        else:
            resnet = get_resnet_model(name)(pretrained=pretrained, progress=False)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            #seq = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            #       resnet.layer1, resnet.layer2, resnet.layer3]
            #self.part_encoder = nn.Sequential(*seq)
        
    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        return output
        #return self.part_encoder(x)

class MultiEncoder(nn.Module):
    color_channels = 3
    
    def __init__(self, img_size, **kwargs):
        super().__init__()
        n_features = kwargs.pop('n_features', 128)
        self.out_ch = n_features
        self.unet = UNet(self.color_channels,n_features,32)
        self.sa = SpatialAttention(kernel_size=7)
        
    def forward(self,x):
        features = self.unet(x) # 64x64x128
        attention = self.sa(features) # 64x64x1
        results = torch.mean(features*attention,(2,3))
        return results

class AttEncoder(nn.Module):
    color_channels = 3
    def __init__(self, img_size, **kwargs):
        super().__init__()
        n_features = kwargs.pop('n_features',128)
        self.out_ch = n_features
        self.unet = UNet(self.color_channels,n_features,32)
        
    def forward(self,x):
        features = self.unet(x) # 64x64x128
        return features

""" 
class AttentionEncoder(nn.Module):
    color_channels = 3
    
    def __init__(self, img_size, name='resnet18',**kwargs):
        super().__init__()
        n_features = kwargs.pop('n_features')
        self.part_encoder = PartEncoder(img_size, **kwargs)
        self.ca = ChannelAttention(channel=256,reduction=16)
        self.sa = SpatialAttention(kernel_size=7)
        out_ch = get_output_size(self.color_channels,img_size,self.part_encoder)
        fc = nn.Sequential()
        if n_features is not None:
            if out_ch != n_features:
                assert n_features < out_ch
                fc = nn.Linear(out_ch, n_features)
                _ = kaiming_weights_init(fc)
                out_ch = n_features # 512
        self.out_ch = out_ch
        self.fc = fc
        
    def forward(self,x):
        output = self.part_encoder(x)
        output = output * self.ca(output)
        results = self.sa(output) * output + output
        return self.fc(results.flatten(1))
"""

class Encoder(nn.Module):
    color_channels = 3

    def __init__(self, img_size, name='resnet18', **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.with_pool = kwargs.pop('with_pool', True)
        pretrained = kwargs.pop('pretrained', False)
        #n_features = kwargs.pop('n_features')
        n_features = kwargs.pop('n_features', None)
        assert len(kwargs) == 0
        if name == 'identity':
            self.encoder = Identity()
        else:
            resnet = get_resnet_model(name)(pretrained=pretrained, progress=False)
            seq = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                   resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
            if self.with_pool:
                size = self.with_pool if isinstance(self.with_pool, (tuple, list)) else (1, 1)
                seq.append(torch.nn.AdaptiveAvgPool2d(output_size=size)) 
            self.encoder = nn.Sequential(*seq)
        out_ch = get_output_size(self.color_channels, img_size, self.encoder)
        fc = nn.Sequential()
        if n_features is not None:
            if out_ch != n_features:
                assert n_features < out_ch
                fc = nn.Linear(out_ch, n_features)
                _ = kaiming_weights_init(fc)
                out_ch = n_features # 512
        self.out_ch = out_ch
        self.fc = fc
        
    def forward(self, x):
        return self.fc(self.encoder(x).flatten(1))
       
class MiniEncoder(nn.Module):
    color_channels = 3

    def __init__(self, img_size, name='resnet18', **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.with_pool = kwargs.pop('with_pool', True)
        pretrained = kwargs.pop('pretrained', False)
        n_features = kwargs.pop('n_features', None)
        assert len(kwargs) == 0
        if name == 'identity':
            self.encoder = Identity()
        else:
            resnet = get_resnet_model(name)(pretrained=pretrained, progress=False)
            seq = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                   resnet.layer1, resnet.layer2]
            if self.with_pool:
                size = self.with_pool if isinstance(self.with_pool, (tuple, list)) else (1, 1)
                seq.append(torch.nn.AdaptiveAvgPool2d(output_size=size)) 
            self.encoder = nn.Sequential(*seq)
        out_ch = get_output_size(self.color_channels, img_size, self.encoder)
        fc = nn.Sequential()
        if n_features is not None:
            if out_ch != n_features:
                assert n_features < out_ch
                fc = nn.Linear(out_ch, n_features)
                _ = kaiming_weights_init(fc)
                out_ch = n_features # 128
        self.out_ch = out_ch
        self.fc = fc
        
    def forward(self, x):
        return self.fc(self.encoder(x).flatten(1))
        
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size,padding=kernel_size//2)
        #self.conv = nn.Conv2d(128,1,kernel_size,padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        max_result,_ = torch.max(x,dim=1,keepdim=True)
        avg_result = torch.mean(x,dim=1,keepdim=True)
        result = torch.cat([max_result,avg_result],1) 
        output = self.conv(result)
        #output = self.conv(x)
        output = self.sigmoid(output)
        return output   

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out+avg_out)
        return output
        
class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weights,mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
    
    def forward(self,x):
        b,c,_,_ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

"""
class PartEncoder(nn.Module):
    def __init__(self,img_size,K,**kwargs):
        super().__init__()
        self.encoder = Encoder(img_size,**kwargs)
    
    def forward(self,x,K):
        outputs = []
        for k in range(K):
            outputs.append(self.encoder(x))
        outputs = torch.stack(outputs)
        return outputs
"""