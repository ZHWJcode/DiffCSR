import os
import torch
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


def count_trainable_parameters(model):
    # numel() returns the total element count.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.1f} million")
    return trainable_params


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def Normalize2(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=False)


def save_heatmap_from_tensor(tensor_map: torch.Tensor,
                             save_path: str,
                             batch_index: int = 0,
                             cmap: str = 'viridis'):
    """
    Average a 4D tensor over channels and save it as a heatmap.

    Args:
        tensor_map (torch.Tensor): Input tensor with shape [B, C, H, W].
        save_path (str): Path to the output PNG file.
        batch_index (int): Batch index to visualize.
        cmap (str): Matplotlib colormap name.
    """

    # Ensure the save directory exists.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Select one batch and move it to CPU.
    # Shape becomes [C, H, W].
    try:
        map_all_channels = tensor_map[batch_index].detach().cpu()
    except IndexError as e:
        print(f"Error: cannot index [B={batch_index}] from {tensor_map.shape}. {e}")
        return

    # Average over all channels.
    map_2d_avg = map_all_channels.mean(dim=0).numpy()

    # Normalize to [0, 1] for the colormap.
    # Add a small epsilon to avoid division by zero.
    map_norm = (map_2d_avg - map_2d_avg.min()) / (map_2d_avg.max() - map_2d_avg.min() + 1e-6)

    # Apply the colormap.
    colormap = plt.get_cmap(cmap)
    heatmap_rgb = colormap(map_norm)[:, :, :3]  # Drop the alpha channel.

    # Convert to uint8 and build a PIL image.
    img_pil = Image.fromarray((heatmap_rgb * 255).astype(np.uint8))

    # Save the image.
    img_pil.save(save_path)
    print(f"Heatmap saved to: {save_path}")


# --------- ResBlock for PSM ---------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = Normalize(self.in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(self.out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_in = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False) if self.in_channels != self.out_channels else nn.Identity()

    def forward(self, x):
        feat = self.norm1(x)
        feat = self.relu1(feat)
        feat = self.conv1(feat)
        feat = self.norm2(feat)
        feat = self.relu2(feat)
        feat = self.conv2(feat)

        residual = self.conv_in(x)

        return feat + residual


# --------- PSM Module ---------
class PSM(nn.Module):
    def __init__(self, guidance_ch=4, target_ch=320, hidden_dim=128):
        """
        Args:
            guidance_ch: Number of guidance input channels.
            target_ch: Base channel count of the target network.
            hidden_dim: Hidden channel size used internally.
        """
        super().__init__()
        self.target_ch = target_ch

        # 1. Shared stem.
        # Map guidance input to the hidden width.
        self.stem = nn.Sequential(
            nn.Conv2d(guidance_ch, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 2. Branch 3 for 64x64 features.
        # Input: [B, 128, 64, 64] -> Output: [B, 640, 64, 64]
        self.head_3 = nn.Sequential(
            ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, target_ch * 2, kernel_size=3, padding=1)
        )

        # 3. Branch 2 for 64x64 features.
        # Input: [B, 128, 64, 64] -> Output: [B, 1280, 64, 64]
        self.down_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.head_2 = nn.Sequential(
            ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, target_ch * 4, kernel_size=3, padding=1)
        )

        # 4. Branch 1 for 32x32 features.
        # Input: [B, 128, 64, 64] -> Output: [B, 2560, 32, 32]
        self.down_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.head_1 = nn.Sequential(
            ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, target_ch * 8, kernel_size=3, padding=1)
        )

        # 5. Zero initialization.
        # Keep the initial modulation close to identity.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Zero-init the output projection layers.
        if isinstance(m, nn.Conv2d):
            # Only reset the final projection layers.
            if m.out_channels in [self.target_ch * 2, self.target_ch * 4, self.target_ch * 8]:
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_guidance):
        # z_guidance shape: [B, 4, 64, 64]
        modulation = {}

        # 1. Extract base features.
        feat_0 = self.stem(z_guidance)

        # 2. Generate modulation for up_layer3.
        params3 = self.head_3(feat_0)  # torch.Size([B, 640, 64, 64])
        scale3, shift3 = params3.chunk(2, dim=1)  # torch.Size([B, 320, 64, 64])
        modulation['scale3'] = scale3
        modulation['shift3'] = shift3

        # 3. Generate modulation for up_layer2.
        feat_2 = self.down_2(feat_0)  # Shape: [B, 128, 64, 64]
        feat_2 = F.relu(feat_2)  # Apply activation.
        params2 = self.head_2(feat_2)  # Conv(128 -> 1280) torch.Size([B, 1280, 64, 64])
        scale2, shift2 = params2.chunk(2, dim=1)  # torch.Size([B, 640, 64, 64])
        modulation['scale2'] = scale2
        modulation['shift2'] = shift2

        # 4. Generate modulation for up_layer1.
        feat_1 = self.down_1(feat_2)  # torch.Size([B, 128, 32, 32])
        feat_1 = F.relu(feat_1)

        params1 = self.head_1(feat_1)  # Conv(128 -> 2560) torch.Size([B, 2560, 32, 32])
        scale1, shift1 = params1.chunk(2, dim=1)  # torch.Size([B, 1280, 32, 32])
        modulation['scale1'] = scale1
        modulation['shift1'] = shift1

        return modulation

if __name__ == "__main__":
    input = torch.randn(2, 4, 64, 64).cuda()
    input = input * 2 - 1
    model = PSM(guidance_ch=4, target_ch=320, hidden_dim=128).cuda()

    output = model(input)
    print(f"PSM's output size is {output.shape}")
