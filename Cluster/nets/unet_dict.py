import torch
import torch.nn as nn
import torch.nn.functional as F
from math import gcd
from nets.pruner import Pruner


class ContractingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, depthwise_conv=False):
        super(ContractingBlock, self).__init__()
        self.depthwise_conv=depthwise_conv
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.layer_dict = nn.ModuleDict()
        self.build_module()


    def build_module(self):
        if self.depthwise_conv:
            groups_1 = gcd(self.out_channels,self.in_channels)
            groups_2 = self.out_channels
        else:
            groups_1 = 1
            groups_2 = 1

        self.layer_dict['skip']=torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.in_channels, out_channels=self.out_channels, groups=groups_1, padding=1)
        self.layer_dict['conv_1']=torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.in_channels, out_channels=self.out_channels, groups=groups_1, padding=1)
        self.layer_dict['relu_1']=torch.nn.ReLU()
        self.layer_dict['bn_1']=torch.nn.BatchNorm2d(self.out_channels)
        if self.depthwise_conv:
            self.layer_dict['conv_1_m'] = torch.nn.Conv2d(kernel_size=1, in_channels=self.out_channels,
                                                        out_channels=self.out_channels, padding=0)
            self.layer_dict['relu_1_m'] = torch.nn.ReLU()
            self.layer_dict['bn_1_m'] = torch.nn.BatchNorm2d(self.out_channels)

        self.layer_dict['conv_2']=torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.out_channels, out_channels=self.out_channels, groups=groups_2, padding=1)
        self.layer_dict['relu_2']=torch.nn.ReLU()
        self.layer_dict['bn_2']=torch.nn.BatchNorm2d(self.out_channels)
        if self.depthwise_conv:
            self.layer_dict['conv_2_m'] = torch.nn.Conv2d(kernel_size=1, in_channels=self.out_channels,
                                                        out_channels=self.out_channels, padding=0)
            self.layer_dict['relu_2_m'] = torch.nn.ReLU()
            self.layer_dict['bn_2_m'] = torch.nn.BatchNorm2d(self.out_channels)


    def forward(self , x ):
        skip = self.layer_dict["skip"](x)
        out = self.layer_dict["conv_1"](x)
        out = self.layer_dict['relu_1'](out)
        out = self.layer_dict['bn_1'](out)
        if self.depthwise_conv:
            out = self.layer_dict["conv_1_m"](out)
            out = self.layer_dict["relu_1_m"](out)
            out = self.layer_dict["bn_1_m"](out)
        out = self.layer_dict['conv_2'](out)
        if self.depthwise_conv:
            out = self.layer_dict['relu_2'](out)
        else:
            out = self.layer_dict['relu_2'](out + skip)
        out = self.layer_dict['bn_2'](out)
        if self.depthwise_conv:
            out = self.layer_dict["conv_2_m"](out)
            out = self.layer_dict["relu_2_m"](out+skip)
            out = self.layer_dict["bn_2_m"](out)
        return out

class Bottleneck(nn.Module):

    def __init__(self, bottle_neck_channels, kernel_size=3, depthwise_conv=False):
        super(Bottleneck, self).__init__()
        self.depthwise_conv = depthwise_conv
        self.bottle_neck_channels = bottle_neck_channels
        self.kernel_size = kernel_size
        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def build_module(self):
        if self.depthwise_conv:
            groups_1 = self.bottle_neck_channels // 2
            groups_2 = self.bottle_neck_channels
        else:
            groups_1 = 1
            groups_2 = 1

        self.layer_dict['skip']=torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.bottle_neck_channels // 2, out_channels=self.bottle_neck_channels, groups=groups_1, padding=1)
        self.layer_dict['conv_1']=torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.bottle_neck_channels // 2, out_channels=self.bottle_neck_channels, groups=groups_1, padding=1)
        self.layer_dict['relu_1']=torch.nn.ReLU()
        self.layer_dict['bn_1']=torch.nn.BatchNorm2d(self.bottle_neck_channels)
        if self.depthwise_conv:
            self.layer_dict['conv_1_m'] = torch.nn.Conv2d(kernel_size=1,
                                                        in_channels=self.bottle_neck_channels,
                                                        out_channels=self.bottle_neck_channels,
                                                        padding=0)
            self.layer_dict['relu_1_m'] = torch.nn.ReLU()
            self.layer_dict['bn_1_m'] = torch.nn.BatchNorm2d(self.bottle_neck_channels)
        self.layer_dict['conv_2']=torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.bottle_neck_channels, out_channels=self.bottle_neck_channels, groups=groups_2, padding=1)
        self.layer_dict['relu_2']=torch.nn.ReLU()
        self.layer_dict['bn_2']=torch.nn.BatchNorm2d(self.bottle_neck_channels)
        if self.depthwise_conv:
            self.layer_dict['conv_2_m'] = torch.nn.Conv2d(kernel_size=1,
                                                        in_channels=self.bottle_neck_channels,
                                                        out_channels=self.bottle_neck_channels,
                                                        padding=0)
            self.layer_dict['relu_2_m'] = torch.nn.ReLU()
            self.layer_dict['bn_2_m'] = torch.nn.BatchNorm2d(self.bottle_neck_channels)
        self.layer_dict['deconv']=torch.nn.ConvTranspose2d(in_channels=self.bottle_neck_channels, out_channels=self.bottle_neck_channels // 2, kernel_size=self.kernel_size, groups=groups_1, stride=2, padding=1, output_padding=1)
        if self.depthwise_conv:
            self.layer_dict['conv_3_m'] = torch.nn.Conv2d(kernel_size=1,
                                                        in_channels=self.bottle_neck_channels//2,
                                                        out_channels=self.bottle_neck_channels//2,
                                                        padding=0)
            self.layer_dict['relu_3_m'] = torch.nn.ReLU()
            self.layer_dict['bn_3_m'] = torch.nn.BatchNorm2d(self.bottle_neck_channels//2)


    def forward(self , x ):
        skip = self.layer_dict["skip"](x)
        out = self.layer_dict["conv_1"](x)
        out = self.layer_dict['relu_1'](out)
        out = self.layer_dict['bn_1'](out)
        if self.depthwise_conv:
            out = self.layer_dict["conv_1_m"](out)
            out = self.layer_dict["relu_1_m"](out)
            out = self.layer_dict["bn_1_m"](out)

        out = self.layer_dict['conv_2'](out)
        if self.depthwise_conv:
            out = self.layer_dict['relu_2'](out)
        else:
            out = self.layer_dict['relu_2'](out + skip)
        out = self.layer_dict['bn_2'](out)
        if self.depthwise_conv:
            out = self.layer_dict["conv_2_m"](out)
            out = self.layer_dict["relu_2_m"](out+skip)
            out = self.layer_dict["bn_2_m"](out)
        out = self.layer_dict['deconv'](out)
        if self.depthwise_conv:
            out = self.layer_dict["conv_3_m"](out)
            out = self.layer_dict["relu_3_m"](out)
            out = self.layer_dict["bn_3_m"](out)
        return out

class ExpansiveBlock(nn.Module):

    def __init__(self, in_channels, mid_channel, out_channels, kernel_size=3,depthwise_conv=False):
        super(ExpansiveBlock, self).__init__()

        self.depthwise_conv = depthwise_conv
        self.in_channels=in_channels
        self.mid_channel = mid_channel
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def build_module(self):
        if self.depthwise_conv:
            groups_1 = 1
            groups_2 = 1
            groups_3 = 1
        else:
            groups_1 = 1
            groups_2 = 1
            groups_3 = 1

        self.layer_dict['skip'] = torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.in_channels, out_channels=self.mid_channel, groups=groups_1, padding=1)
        self.layer_dict['conv_1']=torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.in_channels, out_channels=self.mid_channel, groups=groups_1, padding=1)
        self.layer_dict['relu_1']=torch.nn.ReLU()
        self.layer_dict['bn_1']=torch.nn.BatchNorm2d(self.mid_channel)
        if self.depthwise_conv:
            self.layer_dict['conv_1_m'] = torch.nn.Conv2d(kernel_size=1,
                                                        in_channels=self.mid_channel,
                                                        out_channels=self.mid_channel,
                                                        padding=0)
            self.layer_dict['relu_1_m'] = torch.nn.ReLU()
            self.layer_dict['bn_1_m'] = torch.nn.BatchNorm2d(self.mid_channel)

        self.layer_dict['conv_2']=torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.mid_channel, out_channels=self.mid_channel, groups=groups_2, padding=1)
        self.layer_dict['relu_2']=torch.nn.ReLU()
        self.layer_dict['bn_2']=torch.nn.BatchNorm2d(self.mid_channel)
        if self.depthwise_conv:
            self.layer_dict['conv_2_m'] = torch.nn.Conv2d(kernel_size=1,
                                                          in_channels=self.mid_channel,
                                                          out_channels=self.mid_channel,
                                                          padding=0)
            self.layer_dict['relu_2_m'] = torch.nn.ReLU()
            self.layer_dict['bn_2_m'] = torch.nn.BatchNorm2d(self.mid_channel)
        self.layer_dict['deconv']=torch.nn.ConvTranspose2d(in_channels=self.mid_channel, out_channels=self.out_channels, kernel_size=3, stride=2, groups=groups_3, padding=1, output_padding=1)
        if self.depthwise_conv:
            self.layer_dict['conv_3_m'] = torch.nn.Conv2d(kernel_size=1,
                                                          in_channels=self.out_channels,
                                                          out_channels=self.out_channels,
                                                          padding=0)
            self.layer_dict['relu_3_m'] = torch.nn.ReLU()
            self.layer_dict['bn_3_m'] = torch.nn.BatchNorm2d(self.out_channels)

    def forward(self , x ):
        skip = self.layer_dict["skip"](x)
        out = self.layer_dict["conv_1"](x)
        out = self.layer_dict['relu_1'](out)
        out = self.layer_dict['bn_1'](out)
        if self.depthwise_conv:
            out = self.layer_dict["conv_1_m"](out)
            out = self.layer_dict["relu_1_m"](out)
            out = self.layer_dict["bn_1_m"](out)

        out = self.layer_dict['conv_2'](out)
        if self.depthwise_conv:
            out = self.layer_dict['relu_2'](out)
        else:
            out = self.layer_dict['relu_2'](out + skip)
        out = self.layer_dict['bn_2'](out)
        if self.depthwise_conv:
            out = self.layer_dict["conv_2_m"](out)
            out = self.layer_dict["relu_2_m"](out+skip)
            out = self.layer_dict["bn_2_m"](out)

        out = self.layer_dict['deconv'](out)
        if self.depthwise_conv:
            out = self.layer_dict["conv_3_m"](out)
            out = self.layer_dict["relu_3_m"](out)
            out = self.layer_dict["bn_3_m"](out)
        return out

class FinalBlock(nn.Module):

    def __init__(self, in_channels, mid_channel, out_channels, kernel_size=3,depthwise_conv=False):
        super(FinalBlock, self).__init__()
        self.depthwise_conv = depthwise_conv
        self.in_channels=in_channels
        self.mid_channel = mid_channel
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def build_module(self):
        if self.depthwise_conv:
            groups_1 = gcd(self.in_channels, self.out_channels)
            groups_2 = gcd(self.in_channels,self.mid_channel)
            groups_3 = self.mid_channel
            groups_4 = gcd(self.mid_channel, self.out_channels)
        else:
            groups_1 = 1
            groups_2 = 1
            groups_3 = 1
            groups_4 = 1
        self.layer_dict['skip'] = torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.in_channels, out_channels=self.out_channels, groups= groups_1, padding=1)
        self.layer_dict['conv_1']=torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.in_channels, out_channels=self.mid_channel, groups= groups_2, padding=1)
        self.layer_dict['relu_1']=torch.nn.ReLU()
        self.layer_dict['bn_1']=torch.nn.BatchNorm2d(self.mid_channel)
        if self.depthwise_conv:
            self.layer_dict['conv_1_m'] = torch.nn.Conv2d(kernel_size=1, in_channels=self.mid_channel,
                                                        out_channels=self.mid_channel, padding=0)
            self.layer_dict['relu_1_m'] = torch.nn.ReLU()
            self.layer_dict['bn_1_m'] = torch.nn.BatchNorm2d(self.mid_channel)

        self.layer_dict['conv_2']=torch.nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.mid_channel, out_channels=self.mid_channel, groups= groups_3, padding=1)
        self.layer_dict['relu_2']=torch.nn.ReLU()
        self.layer_dict['bn_2']=torch.nn.BatchNorm2d(self.mid_channel)
        if self.depthwise_conv:
            self.layer_dict['conv_2_m'] = torch.nn.Conv2d(kernel_size=1, in_channels=self.mid_channel,
                                                        out_channels=self.mid_channel, padding=0)
            self.layer_dict['relu_2_m'] = torch.nn.ReLU()
            self.layer_dict['bn_2_m'] = torch.nn.BatchNorm2d(self.mid_channel)
        self.layer_dict['conv_3'] = torch.nn.Conv2d(kernel_size=1, in_channels=self.mid_channel, out_channels=self.out_channels, groups= groups_4)
        self.layer_dict['relu_3']=torch.nn.ReLU()
        self.layer_dict['bn_3']=torch.nn.BatchNorm2d(self.out_channels)
        if self.depthwise_conv:
            self.layer_dict['conv_3_m'] = torch.nn.Conv2d(kernel_size=1, in_channels=self.out_channels,
                                                        out_channels=self.out_channels, padding=0)
            self.layer_dict['relu_3_m'] = torch.nn.ReLU()
            self.layer_dict['bn_3_m'] = torch.nn.BatchNorm2d(self.out_channels)

    def forward(self , x ):
        skip = self.layer_dict["skip"](x)
        out = self.layer_dict["conv_1"](x)
        out = self.layer_dict['relu_1'](out)
        out = self.layer_dict['bn_1'](out)
        if self.depthwise_conv:
            out = self.layer_dict["conv_1_m"](out)
            out = self.layer_dict["relu_1_m"](out)
            out = self.layer_dict["bn_1_m"](out)
        out = self.layer_dict['conv_2'](out)
        out = self.layer_dict['relu_2'](out)
        out = self.layer_dict['bn_2'](out)
        if self.depthwise_conv:
            out = self.layer_dict["conv_2_m"](out)
            out = self.layer_dict["relu_2_m"](out)
            out = self.layer_dict["bn_2_m"](out)
        out = self.layer_dict['conv_3'](out)
        if self.depthwise_conv:
            out = self.layer_dict['relu_3'](out)
        else:
            out = self.layer_dict['relu_3'](out + skip)
        out = self.layer_dict['bn_3'](out)
        if self.depthwise_conv:
            out = self.layer_dict["conv_3_m"](out)
            out = self.layer_dict["relu_3_m"](out+skip)
            out = self.layer_dict["bn_3_m"](out)
        return out

class UNetDict(nn.Module):
    def contracting_block_seq(self, in_channels, out_channels, kernel_size=3, depthwise_conv=False):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block


    def expansive_block_seq(self, in_channels, mid_channel, out_channels, kernel_size=3, depthwise_conv=False):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        return block

    def bottle_neck_seq(self, bottle_neck_channels, depthwise_conv=False):

        block= torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=bottle_neck_channels//2, out_channels=bottle_neck_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(bottle_neck_channels),
            torch.nn.Conv2d(kernel_size=3, in_channels=bottle_neck_channels, out_channels=bottle_neck_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(bottle_neck_channels),
            torch.nn.ConvTranspose2d(in_channels=bottle_neck_channels, out_channels=bottle_neck_channels//2, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        )
        return  block



    def final_block_seq(self, in_channels, mid_channel, out_channels, kernel_size=3, depthwise_conv=False):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=1, in_channels=mid_channel, out_channels=out_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def __init__(self, in_channel, out_channel, hour_glass_depth, bottle_neck_channels,use_skip, prune_prob=0, depthwise_conv=False):
        super(UNetDict, self).__init__()
        # Encode
        self.hour_glass_depth=hour_glass_depth
        self.bottle_neck_channels=bottle_neck_channels
        self.layer_dict = nn.ModuleDict()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_skip = use_skip
        self.depthwise_conv=depthwise_conv
        self.prune_prob=prune_prob
        # build the network
        if self.use_skip:
            self.contracting_block = ContractingBlock
            self.bottle_neck = Bottleneck
            self.expansive_block = ExpansiveBlock
            self.final_block = FinalBlock
        else:
            self.contracting_block = self.contracting_block_seq
            self.bottle_neck = self.bottle_neck_seq
            self.expansive_block = self.expansive_block_seq
            self.final_block = self.final_block_seq

        self.build_module()
        if prune_prob !=0:
            self.pruner=Pruner(self.layer_dict, self.prune_prob)

    def build_module(self):
        self.layer_dict[f"conv_encode1"]  = self.contracting_block(in_channels=self.in_channel, out_channels=self.bottle_neck_channels//2**(self.hour_glass_depth), depthwise_conv=self.depthwise_conv)
        self.layer_dict[f"conv_maxpool1"] = torch.nn.MaxPool2d(kernel_size=2)
        for i in range(2,self.hour_glass_depth+1):
            self.layer_dict[f"conv_encode{i}"] = self.contracting_block(in_channels=self.bottle_neck_channels//2**(self.hour_glass_depth-i+2), out_channels=self.bottle_neck_channels//2**(self.hour_glass_depth-i+1), depthwise_conv=self.depthwise_conv)
            self.layer_dict[f"conv_maxpool{i}"] = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.layer_dict[f"bottleneck"] = self.bottle_neck(bottle_neck_channels=self.bottle_neck_channels, depthwise_conv=self.depthwise_conv)
        # Decode
        for i in range(self.hour_glass_depth,1,-1):
            self.layer_dict[f"conv_decode{i}"] = self.expansive_block(in_channels=self.bottle_neck_channels//2**(self.hour_glass_depth-i), mid_channel=self.bottle_neck_channels//2**(self.hour_glass_depth-i+1), out_channels= self.bottle_neck_channels//2**(self.hour_glass_depth-i+2), depthwise_conv=self.depthwise_conv)
        self.layer_dict[f"final_layer"] = self.final_block(in_channels=self.bottle_neck_channels//2**(self.hour_glass_depth-1), mid_channel=self.bottle_neck_channels//2**(self.hour_glass_depth), out_channels=self.out_channel, depthwise_conv=self.depthwise_conv)


    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        all={}
        all["conv_encode1_out"] = self.layer_dict[f"conv_encode1"](x)
        for i in range(1,self.hour_glass_depth):
            out = self.layer_dict[f"conv_maxpool{i}"](all[f"conv_encode{i}_out"])
            all[f"conv_encode{i+1}_out"] = self.layer_dict[f"conv_encode{i+1}"](out)
        out = self.layer_dict[f"conv_maxpool{self.hour_glass_depth}"](all[f"conv_encode{self.hour_glass_depth}_out"])
        # Bottleneck
        out =self.layer_dict[f"bottleneck"](out)
        # Decode
        out = self.crop_and_concat(out, all[f"conv_encode{self.hour_glass_depth}_out"], crop=False)
        for i in range(self.hour_glass_depth,1,-1):
            out = self.layer_dict[f"conv_decode{i}"](out)
            out = self.crop_and_concat(out, all[f"conv_encode{i-1}_out"], crop=False)
        final_layer = self.layer_dict[f"final_layer"](out)
        out = final_layer.permute(0, 2, 3, 1)

        return out


    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        # with torch.no_grad():
        #     self.layer_dict["bottleneck"].layer_dict["conv_1"].weight[0,0,:,:] *=  torch.Tensor([[1 ,0, -1],[2, 0 ,-2], [1, 0 ,-1]])
        # print (self.layer_dict["bottleneck"].layer_dict["conv_1"].weight[0,0,1,1])
        # print (type(self.layer_dict["bottleneck"].layer_dict["conv_1"].weight[0,0,1,1]))
        return
