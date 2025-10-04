"""

References:
[1]: https://arxiv.org/abs/1802.10508v1
    3D Improved UNet Paper
[2]: https://arxiv.org/abs/1603.05027
    Pre-activation residual block

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextModule(nn.Module):
    def __init__(self, num_features, kernel_size=3, p_dropout=0.3):
        super().__init__()

        # as per [1], replace BatchNorm with InstanceNorm
        self.path = nn.Sequential(
            nn.InstanceNorm3d(num_features),
            nn.Conv3d(num_features, num_features, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.01),
            nn.Dropout3d(p_dropout),
            nn.InstanceNorm3d(num_features),
            nn.Conv3d(num_features, num_features, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.path(x) + x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pre_kernel=3, context_kernel=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=pre_kernel, stride=2),
            ContextModule(out_channels, kernel_size=context_kernel)
        )
    
    def forward(self, x):
        return self.blocks(x)
    
class UpscaleModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.blocks(x)

class LocalisationModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size),
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        return self.blocks(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upscale = UpscaleModule(in_channels, out_channels)
        self.localisation = LocalisationModule(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.upscale(x)
        # concatenate along the channel dimension then pass to
        # localisation module
        # the paper doesn't specify how information is copied
        # TODO: ask about this
        resized = F.interpolate(y, x.shape[2:])
        res = self.localisation(torch.cat([resized, x], dim=1))
        return res
    
class Improved3DUnet(nn.Module):
    def __init__(self, n_classes, initial_channels, depth):
        super().__init__()
        self.initial_block = nn.Sequential(
            nn.Conv3d(1, initial_channels, kernel_size=3),
            ContextModule(initial_channels, kernel_size=3)
        )

        self.down_blocks = []
        for i in range(depth):
            c_in = initial_channels * 2**i
            c_out = 2 * c_in
            block = DownBlock(c_in, c_out, pre_kernel=3, context_kernel=3)
            self.down_blocks.append(block)
            self.add_module(f"Down Block {i+1}", block)

        self.up_blocks = []
        self.segmentation_layers = []
        for i in range(depth-1,-1,-1):
            c_in = initial_channels * 2**i * 2
            c_out = c_in // 2
            block = UpBlock(c_in, c_out)
            self.up_blocks.append(block)
            self.add_module(f"Up Block {i+1}", block)
            if i < depth - 1:
                # i.e. for all layers except the first up block
                # TODO: ask about kernel size
                seg = nn.Conv3d(c_out, n_classes, kernel_size=1)
                self.segmentation_layers.append(seg)
                self.add_module(f"Segmentation for UpBlock {i+1}", seg)
        
        
    def forward(self, x):

        initial_block_result = self.initial_block(x)

        down_layer_outputs = []

        down_layer_outputs.append(initial_block_result)

        for i, down_block in enumerate(self.down_blocks):
            input_data = down_layer_outputs[i]
            y = down_block(input_data)
            down_layer_outputs.append(y)

        # y holds the final down layer output
        up_layer_outputs = []
        for i, up_block in enumerate(self.up_blocks):
            if i == 0:
                input_data = y
            else:
                input_data = up_layer_outputs[i-1]
            reversed_index = len(self.up_blocks) - 1 - i
            up_layer_outputs.append(up_block(input_data, down_layer_outputs[reversed_index]))
        
        # apply segmentation layers to reduce channels to feature maps

        for i, seg in enumerate(self.segmentation_layers):
            # since the deepest localisation output is not used
            up_index = i + 1
            if i == 0:
                output = seg(up_layer_outputs[up_index])
            else:
                # also combine with previous output
                current_output: torch.Tensor = seg(up_layer_outputs[up_index])
                output = current_output + F.interpolate(output, current_output.shape[2:])
        
        return F.softmax(output, dim=1)

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = Improved3DUnet(10, 16, 4).to(device)
    dummy = torch.rand(1,1,256,256,128).to(device)
    print(net(dummy).shape)
