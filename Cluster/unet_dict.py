import torch
import torch.nn as nn
import torch.nn.functional as F



class UNetDict(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
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

    def bottle_neck(self, bottle_neck_channels):

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



    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
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

    def __init__(self, in_channel, out_channel, hour_glass_depth, bottle_neck_channels):
        super(UNetDict, self).__init__()
        # Encode
        self.hour_glass_depth=hour_glass_depth
        self.bottle_neck_channels=bottle_neck_channels
        self.layer_dict = nn.ModuleDict()
        self.in_channel = in_channel
        self.out_channel = out_channel
        # build the network
        self.build_module()

    def build_module(self):
        self.layer_dict[f"conv_encode1"]  = self.contracting_block(in_channels=self.in_channel, out_channels=self.bottle_neck_channels//2**(self.hour_glass_depth))
        self.layer_dict[f"conv_maxpool1"] = torch.nn.MaxPool2d(kernel_size=2)
        for i in range(2,self.hour_glass_depth+1):
            self.layer_dict[f"conv_encode{i}"] = self.contracting_block(in_channels=self.bottle_neck_channels//2**(self.hour_glass_depth-i+2), out_channels=self.bottle_neck_channels//2**(self.hour_glass_depth-i+1))
            self.layer_dict[f"conv_maxpool{i}"] = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.layer_dict[f"bottleneck"] = self.bottle_neck(bottle_neck_channels=self.bottle_neck_channels)
        # Decode
        for i in range(self.hour_glass_depth,1,-1):
            self.layer_dict[f"conv_decode{i}"] = self.expansive_block(in_channels=self.bottle_neck_channels//2**(self.hour_glass_depth-i), mid_channel=self.bottle_neck_channels//2**(self.hour_glass_depth-i+1), out_channels= self.bottle_neck_channels//2**(self.hour_glass_depth-i+2))
        self.layer_dict[f"final_layer"] = self.final_block(in_channels=self.bottle_neck_channels//2**(self.hour_glass_depth-1), mid_channel=self.bottle_neck_channels//2**(self.hour_glass_depth), out_channels=self.out_channel)


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
        outputs = final_layer.permute(0, 2, 3, 1)
        return outputs

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        return
