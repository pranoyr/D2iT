import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from diffusion.model import ResnetBlock, AttnBlock, Downsample, Upsample, Normalize, nonlinearity
import pytorch_lightning as pl
import numpy as np


# encoderconfig:
#       target: modules.dynamic_modules.EncoderDual.DualGrainEncoder
#       params:
#         ch: 128
#         ch_mult: [1,1,2,2,4]
#         num_res_blocks: 2
#         attn_resolutions: [16, 32]
#         dropout: 0.0
#         resamp_with_conv: true
#         in_channels: 3
#         resolution: 256
#         z_channels: 256
#         router_config:
#           target: modules.dynamic_modules.RouterDual.DualGrainFeatureRouter
#           params:
#             num_channels: 256
#             normalization_type: group-32
#             gate_type: 2layer-fc-SiLu
#     decoderconfig:
#       target: modules.dynamic_modules.DecoderPositional.Decoder
#       params:
#         ch: 128
#         in_ch: 256
#         out_ch: 3
#         ch_mult: [1,1,2,2]
#         num_res_blocks: 2
#         resolution: 256
#         attn_resolutions: [32]
#         latent_size: 32
#         window_size: 2
#         position_type: fourier+learned


class DualGrainEncoder(pl.LightningModule):
    def __init__(self, 
        *, 
        ch=128,
        ch_mult=(1,1,2,2,4),
        num_res_blocks=2,
        attn_resolutions=(16, 32),
        dropout=0.0,
        resamp_with_conv=True, 
        in_channels=3,
        resolution=256,
        z_channels=4
        ):
        super().__init__()
        
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle for the coarse grain
        self.mid_coarse = nn.Module()
        self.mid_coarse.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid_coarse.attn_1 = AttnBlock(block_in)
        self.mid_coarse.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # end for the coarse grain
        self.norm_out_coarse = Normalize(block_in)
        self.conv_out_coarse = torch.nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)
        
        block_in_finegrain = block_in // (ch_mult[-1] // ch_mult[-2])
        # middle for the fine grain
        self.mid_fine = nn.Module()
        self.mid_fine.block_1 = ResnetBlock(in_channels=block_in_finegrain, out_channels=block_in_finegrain, temb_channels=self.temb_ch, dropout=dropout)
        self.mid_fine.attn_1 = AttnBlock(block_in_finegrain)
        self.mid_fine.block_2 = ResnetBlock(in_channels=block_in_finegrain, out_channels=block_in_finegrain, temb_channels=self.temb_ch, dropout=dropout)

        # end for the fine grain
        self.norm_out_fine = Normalize(block_in_finegrain)
        self.conv_out_fine = torch.nn.Conv2d(block_in_finegrain, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            if i_level == self.num_resolutions-2:
                h_fine = h

        h_coarse = hs[-1]

        # middle for the h_coarse
        h_coarse = self.mid_coarse.block_1(h_coarse, temb)
        h_coarse = self.mid_coarse.attn_1(h_coarse)
        h_coarse = self.mid_coarse.block_2(h_coarse, temb)

        # end for the h_coarse
        h_coarse = self.norm_out_coarse(h_coarse)
        h_coarse = nonlinearity(h_coarse)
        h_coarse = self.conv_out_coarse(h_coarse)

        # middle for the h_fine
        h_fine = self.mid_fine.block_1(h_fine, temb)
        h_fine = self.mid_fine.attn_1(h_fine)
        h_fine = self.mid_fine.block_2(h_fine, temb)

        # end for the h_fine
        h_fine = self.norm_out_fine(h_fine)
        h_fine = nonlinearity(h_fine)
        h_fine = self.conv_out_fine(h_fine)

        # Just return the individual latents
        return [h_fine, h_coarse]



def compute_entropy_map(image_gray, patch_size=16, num_bins=256, sigma=0.01):
    """
    Optimized entropy map computation using einops and vectorized torch operations.
    
    Paper Algorithm: Gaussian kernel density estimation + Shannon entropy
    
    Args:
        image_gray: [B, H, W] grayscale image tensor
        patch_size: Size of each patch (S in paper)
        num_bins: Number of histogram bins for PDF estimation
        sigma: Gaussian kernel smoothing parameter (paper uses σ = 0.01)
    
    Returns:
        entropy_map: [B, H//patch_size, W//patch_size] entropy values
    """
    B, H, W = image_gray.shape
    device = image_gray.device
    
    # Ensure dimensions are divisible by patch_size
    H_pad = ((H + patch_size - 1) // patch_size) * patch_size
    W_pad = ((W + patch_size - 1) // patch_size) * patch_size
    
    if H_pad != H or W_pad != W:
        image_gray = F.pad(image_gray, (0, W_pad - W, 0, H_pad - H), mode='reflect')
        H, W = H_pad, W_pad
    
    # Reshape into patches using einops
    # [B, H, W] -> [B, H//S, W//S, S, S] -> [B, H//S, W//S, S*S]
    patches = rearrange(
        image_gray, 
        'b (h_patches patch_h) (w_patches patch_w) -> b h_patches w_patches (patch_h patch_w)',
        patch_h=patch_size, 
        patch_w=patch_size
    )
    
    B, H_patches, W_patches, patch_area = patches.shape
    
    # Create histogram bins - uniformly distributed as mentioned in paper
    bins = torch.linspace(0.0, 1.0, num_bins, device=device)
    
    # Vectorized Gaussian kernel density estimation
    # Reshape for broadcasting: patches [B, H_p, W_p, S²] and bins [num_bins]
    patches_expanded = patches.unsqueeze(-1)  # [B, H_p, W_p, S², 1]
    bins_expanded = bins.view(1, 1, 1, 1, num_bins)  # [1, 1, 1, 1, num_bins]
    
    # Paper equation (2): ρ̂_k(b_j) = (1/S²) Σ exp(-1/2 * ((x_k,i - b_j)/σ)²)
    gaussian_weights = torch.exp(-0.5 * ((patches_expanded - bins_expanded) / sigma) ** 2)
    
    # Average over patch pixels (1/S² * sum) -> [B, H_p, W_p, num_bins]
    pdf_values = reduce(gaussian_weights, 'b h w pixels bins -> b h w bins', 'mean')
    
    # Normalize to ensure valid probability distribution
    pdf_values = pdf_values / (pdf_values.sum(dim=-1, keepdim=True) + 1e-10)
    
    # Shannon entropy computation (Equation 3): E_k = -Σ ρ̂_k(b_j) log ρ̂_k(b_j)
    # Add small epsilon to prevent log(0)
    pdf_values = torch.clamp(pdf_values, min=1e-10)
    entropy_map = -torch.sum(pdf_values * torch.log2(pdf_values), dim=-1)
    
    return entropy_map




def assign_grain(entropy_map, grained_ratios=[0.5]):
    """
    Paper method: "pre-calculate entropy distribution of natural images in ImageNet dataset"
    "establish entropy thresholds corresponding to specific percentiles"
    """
    # Paper approach: use percentile-based thresholds
    entropy_flat = entropy_map.flatten()
    
    # Paper mentions r = {r1, r2, ..., rk} grained ratios
    # For dual-grained (k=2), use 50th percentile as threshold
    threshold = torch.quantile(entropy_flat, 1 - grained_ratios[0])
    
    grain_map = torch.zeros_like(entropy_map)
    grain_map[entropy_map > threshold] = 1  # fine-grained regions
    
    return grain_map


def route_latents_paper(latents_fine, latents_coarse, grain_map):
    """
    Paper method: "neighbor copying method" for irregular latent codes
    """
    B, C, Hf, Wf = latents_fine.shape
    
    # Paper: "latent code for each region is copied to the finest granularity 
    # if the finest granularity is not assigned for it"
    coarse_upsampled = F.interpolate(latents_coarse, size=(Hf, Wf), mode='nearest')
    
    # Expand grain_map to match latent dimensions
    grain_map_expanded = repeat(
        grain_map, 
        'b h w -> b c (h h_scale) (w w_scale)', 
        c=C, 
        h_scale=Hf // grain_map.shape[1],
        w_scale=Wf // grain_map.shape[2]
    )
    
    # Paper routing: direct selection based on grain_map
    routed = grain_map_expanded * latents_fine + (1 - grain_map_expanded) * coarse_upsampled
    
    return routed



class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, *wargs, **kwargs):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h





class DVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder =  DualGrainEncoder(
            ch=128,
            ch_mult=(1,1,2,2,4),
            num_res_blocks=2,
            attn_resolutions=(16, 32),
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=3,
            resolution=256,
            z_channels=256
        )



        self.decoder = Decoder(
            ch=128,
            out_ch=3,
            in_channels=256,
            ch_mult=(1,1,2,2),
            num_res_blocks=2,
            attn_resolutions=(32,),
            dropout=0.0,
            resolution=256,
            z_channels=256,
        )



        
    def encode(self, x):
        """Encoding phase - returns routed latents and grain map"""
        # Paper: "hierarchical encoder to encode different image regions 
        # at different downsampling rates"
        latents = self.encoder(x)  # [fine, coarse]


        
        # Paper: convert to single-channel for entropy calculation
        x_gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        
        # Paper algorithm: Dynamic Grained Coding
        entropy_map = compute_entropy_map(x_gray, patch_size=16)
        grain_map = assign_grain(entropy_map, grained_ratios=[0.5])
        
        # Paper: neighbor copying method
        routed_latents = route_latents_paper(latents[0], latents[1], grain_map)
        
        return routed_latents, grain_map, entropy_map
    
    def decode(self, z):
        """Decoding phase - reconstruct image from latents"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode then decode"""
        routed_latents, grain_map, entropy_map = self.encode(x)
        reconstructed = self.decode(routed_latents)
        
        return {
            'reconstructed': reconstructed,
            'latents': routed_latents, 
            'grain_map': grain_map,
            'entropy_map': entropy_map
        }


def vae_loss(x_recon, x_orig, reduction='mean'):
    """
    Simple reconstruction loss for VAE
    In a full VAE, you'd also have KL divergence term
    """
    mse_loss = F.mse_loss(x_recon, x_orig, reduction=reduction)
    return mse_loss


if __name__ == "__main__":
    # Create model
    model = DVAE()  # Smaller for testing
    
    # Test input (normalized to [-1, 1] range for tanh decoder)
    x = torch.randn(2, 3, 256, 256).clamp(-1, 1)
    
    print("Testing DVAE...")
    print("Input shape:", x.shape)
    
    # Test encoding only
    with torch.no_grad():
        latents, grain_map, entropy_map = model.encode(x)
        
    print("\nEncoding Results:")
    print("Latents shape:", latents.shape)           # Should be [2, 4, 32, 32]
    print("Grain Map shape:", grain_map.shape)       # Should be [2, 16, 16]
    print("Entropy Map shape:", entropy_map.shape)   # Should be [2, 16, 16]
    print("Fine regions:", (grain_map == 1).sum().item())
    print("Coarse regions:", (grain_map == 0).sum().item())
    
    # Test full reconstruction
    with torch.no_grad():
        results = model(x)
        
    print("\nFull DVAE Results:")
    print("Reconstructed shape:", results['reconstructed'].shape)  # Should be [2, 3, 256, 256]
    
    # Test reconstruction quality
    recon_loss = vae_loss(results['reconstructed'], x)
    print("Reconstruction Loss:", recon_loss.item())
    
    # Test individual components
    print("\nTesting individual decode:")
    with torch.no_grad():
        decoded = model.decode(latents)
    print("Decoded shape:", decoded.shape)
    print("Fine regions:", (grain_map == 1).sum().item())
    print("Coarse regions:", (grain_map == 0).sum().item())