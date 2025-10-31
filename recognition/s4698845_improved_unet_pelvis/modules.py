"""
Contains classes to build the 3D Improved UNet as described in [1].
When executed standalone, this file checks the input and output dimensions
by passing dummy data to the network. The pre-activation residual
block from [2] is used in the context module from [1].

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
    """Class to represent the Context Module from [1]"""
    def __init__(self, num_channels, kernel_size=3, p_dropout=0.0):
        """Create a new Context module with the given parameters.

        Args:
            num_channels (int): number of channels for input and output
            kernel_size (int, optional): kernel size for convolutional layers.
                Defaults to 3.
            p_dropout (float, optional): dropout layer probability. Defaults to 0.0.
        """
        super().__init__()

        # as per [1], replace BatchNorm with InstanceNorm
        self.path = nn.Sequential(
            nn.InstanceNorm3d(num_channels),
            nn.Conv3d(num_channels, num_channels, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.01),
            nn.Dropout3d(p_dropout),
            nn.InstanceNorm3d(num_channels),
            nn.Conv3d(num_channels, num_channels, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        """Forward propagation, including residual connection.

        Args:
            x (torch.Tensor): batched input

        Returns:
            torch.Tensor: batched output
        """
        return self.path(x) + x


class DownBlock(nn.Module):
    """
    Class to represent a block in the down sampling path of the 3D
    UNet - i.e. the combination of a 3D convolution followed by a context
    module.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 pre_kernel: int = 3, context_kernel: int = 3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=pre_kernel, stride=2, padding=1),
            ContextModule(out_channels, kernel_size=context_kernel)
        )
    
    def forward(self, x: torch.Tensor):
        """Forward propagation through this DownBlock.

        Args:
            x (torch.Tensor): batched input

        Returns:
            torch.Tensor: batched output
        """
        return self.blocks(x)
    
class UpscaleModule(nn.Module):
    """
    Class to represent the upsampling module from [1], which is a direct 2x
    upsample followed by a 3D convolution layer.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3):
        """Create a new Upscale Module with the given parameters.

        Args:
            in_channels (int): number of channel in input
            out_channels (int): number of channels in output
            kernel_size (int, optional): kernel size for convolutional layer.
                Defaults to 3.
        """
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        )

    def forward(self, x: torch.Tensor):
        """Forward propagation through this Upscale Module.

        Args:
            x (torch.Tensor): batched input

        Returns:
            torch.Tensor: batched output
        """
        return self.blocks(x)

class LocalisationModule(nn.Module):
    """
    Class to represent the localisation module from [1], which consists
    of two convolutional layers, with the second only serving to set
    the number of feature maps.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3):
        """Create a new Localisation module with the given parameters.

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            kernel_size (int, optional): kernel size of first convolution layer.
                Defaults to 3.
        """
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor):
        """Forward propagation through this LocalisationModule.

        Args:
            x (torch.Tensor): batched input

        Returns:
            torch.Tensor: batched output
        """
        return self.blocks(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upscale = UpscaleModule(in_channels, out_channels)

        # this also has an input size of 'in_channels' since it will take in the
        # concatenation of the upscaled image and the output of a skip connection
        # since 'in_channels' will be 2 * out_channels
        self.localisation = LocalisationModule(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Calculate ouput of this UpBlock for a given input, and skip
        connection input.

        Args:
            x (torch.Tensor): main input
            y (torch.Tensor): skip connection input from corresponding down block

        Returns:
            torch.Tensor: tensor result
        """
        x = self.upscale(x)
        # concatenate along the channel dimension then pass to
        # localisation module
        resized = F.interpolate(y, x.shape[2:])
        res = self.localisation(torch.cat([resized, x], dim=1))
        return res

class AbstractNetwork(nn.Module):
    # number of classes that this model can predict
    n_classes: int

    @staticmethod
    def loss(predictions: torch.Tensor, labels: torch.Tensor):
        """Calculate value of training loss function for this model. 
        Operates on batched inputs and labels.

        Args:
            predictions (torch.Tensor): network output
            labels (torch.Tensor): labelled images
        """
        pass

    def metric(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Calculate potentially non-differentiable evaluation
        metric. Operates on batched inputs and labels.

        Args:
            outputs (torch.Tensor): network output
            labels (torch.Tensor): labelled images
        """
        pass

class Improved3DUnet(AbstractNetwork):
    """
    Class to represent the Improved 3D UNet structure from [1].
    """
    def __init__(self, n_classes: int, initial_channels: int, depth: int):
        """Create a new Improved3DUnet.

        Args:
            n_classes (int): number of classes to be segmented.
            initial_channels (int): initial number of channels to split into
            depth (int): number of down (and up) steps to have in the network.
        """

        super().__init__()
        self.n_classes = n_classes
        self.initial_block = nn.Sequential(
            nn.Conv3d(1, initial_channels, kernel_size=3, padding=1),
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
                # apply a final conv3D to set the number of output channels
                # to the number of classes so that the output is one-hot encoded
                seg = nn.Conv3d(c_out, n_classes, kernel_size=1)
                self.segmentation_layers.append(seg)
                self.add_module(f"Segmentation for UpBlock {i+1}", seg)
        
        
    def forward(self, x: torch.Tensor):
        """
        Propagate input through the UNet to produce segmentations.
        The number of output channels is the same as `self.n_classes` since
        the segmentations are one-hot encoded.

        The input is required to be batched, so if it is desired to process
        a single image, an extra initial dimension must be added, for example
        `x = single_image.unsqueeze(0)`

        Args:
            x (torch.Tensor): input images (batched)

        Returns:
            torch.Tensor: output segmentations (one-hot encoded)
        """
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
        # 'deep supervision' layers combine final output with output from
        # localisation modules to improve gradient flow

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
    
    @staticmethod
    def loss(predictions: torch.Tensor, labels: torch.Tensor):
        """
        Compute the dice loss as described in equation 1 of [1].
        The input is taken to be of shape (B,C,D,H,W), with C being
        the number of classes. Labels is taken to have shape (B,1,D,H,W)
        with entries being integers from 0 to C-1, representing each class.
        """
        K = predictions.shape[1]
        # squeeze to remove the channel dim (which is 1), as we will replace this
        # with the class one-hot encoding dimension
        one_hot_labels = F.one_hot(labels.squeeze(dim=1)).permute(0,4,1,2,3)
        # one_hot_labels.shape == [B,C,D,H,W]

        # sum over all spatial dimensions (equivalent to summing over the voxels
        # as done in the paper)
        numerator = torch.sum(one_hot_labels * predictions, dim=(2,3,4))
        X = torch.sum(predictions, dim=(2,3,4))
        Y = torch.sum(one_hot_labels, dim=(2,3,4))
        denominator = X + Y
        
        batches = predictions.shape[0]

        # sum over all remaining dimensions, i.e. class and batches
        return -2 / K / batches * torch.sum(numerator / denominator)
    
    def metric(self, outputs: torch.Tensor,
               labels: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the components of the hard (non-differentiable)
        dice score for each class based on the given network outputs
        and ground truth labels.

        Args:
            outputs (torch.Tensor): network outputs
            labels (torch.Tensor): ground truth labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tensors containing the
                interesction count, output mask count and input mask count for each class.
                (these are returned separately so the score can be calculated over a number
                of samples)
        """
        predicted_classes = torch.argmax(outputs, dim=1)

        # the one hots have the class as the last dimension
        one_hot_output = F.one_hot(predicted_classes, self.n_classes)
        one_hot_true = F.one_hot(labels.squeeze(dim=1), self.n_classes)

        XY = (one_hot_output * one_hot_true).sum(dim=(0,1,2,3))
        X = one_hot_true.sum(dim=(0,1,2,3))
        Y = one_hot_output.sum(dim=(0,1,2,3))
        return XY, X, Y


if __name__ == "__main__":
    """
    Test code to veriying the input and output dimensions of the
    Improved 3D UNet model.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    classes = 6
    net = Improved3DUnet(classes, 16, 4).to(device)
    print(net)
    dummy = torch.rand(1,1,256,256,128).to(device)
    dummy_labels = torch.randint_like(dummy, low=0, high=classes)
    
    dummy_prediction = net(dummy)
    print(dummy_prediction.shape)
    print(dummy_labels.shape)
    resized_labels = F.interpolate(dummy_labels, dummy.shape[2:]).long()
    print(net.loss(dummy, resized_labels))
