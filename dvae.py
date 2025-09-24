import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


class SimpleDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor):
        super().__init__()
        # The paper mentions using downsampling factors directly
        # This is a simplified version based on the paper's description
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=3, 
                             stride=downsample_factor, 
                             padding=1)
        
    def forward(self, x):
        return self.conv(x)


class HierarchicalEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, downsample_factors=[8, 16]):
        super().__init__()
        self.downsample_factors = downsample_factors
        
        # Paper mentions: "hierarchical encoder to encode different image regions 
        # at different downsampling rates"
        # Creates separate encoders for each downsampling factor
        self.encoders = nn.ModuleList([
            SimpleDownsampleBlock(in_channels, latent_channels, factor) 
            for factor in downsample_factors
        ])
        
    def forward(self, x):
        # Paper states: "encode image regions into different downsampling 
        # continuous latent representations"
        latents = []
        for encoder in self.encoders:
            latents.append(encoder(x))
        
        # Returns list: [fine_latents (8x), coarse_latents (16x)]
        return latents

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
    def __init__(self, latent_channels=4, out_channels=3, base_channels=128):
        super().__init__()
        
        # Decoder mirrors the encoder structure in reverse
        self.init_conv = nn.Conv2d(latent_channels, base_channels*2, kernel_size=3, padding=1)
        
        # Upsampling blocks to reconstruct from 32x32 -> 256x256 (8x upsampling)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32->64
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64->128
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128->256
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Final output layer
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, z):
        """
        Decode latent representation back to image
        Args:
            z: Latent tensor [B, latent_channels, 32, 32]
        Returns:
            Reconstructed image [B, 3, 256, 256]
        """
        h = self.init_conv(z)  # [B, 256, 32, 32]
        h = self.up1(h)        # [B, 256, 64, 64]
        h = self.up2(h)        # [B, 128, 128, 128]
        h = self.up3(h)        # [B, 128, 256, 256]
        
        # Output with sigmoid/tanh activation depending on input range
        out = torch.tanh(self.output_conv(h))  # [B, 3, 256, 256]
        return out


class DVAE(nn.Module):
    def __init__(self, downsample_factors=[8, 16], base_channels=128):
        super().__init__()
        self.encoder = HierarchicalEncoder(downsample_factors=downsample_factors)
        self.decoder = Decoder(base_channels=base_channels)
        self.downsample_factors = downsample_factors
        
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
    model = DVAE(downsample_factors=[8, 16], base_channels=64)  # Smaller for testing
    
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