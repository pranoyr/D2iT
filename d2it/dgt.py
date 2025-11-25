import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed

# ==========================================
# 1. Helper Components (DiT-S Blocks)
# ==========================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

# ==========================================
# 2. Dynamic Grain Transformer (Main)
# ==========================================

class DynamicGrainTransformer(nn.Module):
    def __init__(
        self,
        input_size=16,       # 16x16 Grain Map
        patch_size=2,        # Standard DiT Patch Size
        in_channels=1,       # 1 Channel for the noisy map
        num_grain_levels=2,  # Binary Classification (Coarse vs Fine)
        hidden_size=384,     # DiT-S
        depth=12,            # DiT-S
        num_heads=6,         # DiT-S
        num_classes=1000,
        num_diffusion_timesteps=1000
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.out_channels = num_grain_levels
        
        # --- DiT Backbone ---
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, num_grain_levels)

        # --- Diffusion Buffers ---
        betas = torch.linspace(1e-4, 0.02, num_diffusion_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.num_timesteps = num_diffusion_timesteps

        self.initialize_weights()

    def initialize_weights(self):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """ Reshapes tokens to (N, num_grain_levels, H, W) logits """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def _run_backbone(self, x, t, y):
        """ Internal helper for the DiT forward pass """
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y)
        c = t + y # Combine Embeddings
        
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)

    def forward(self, target_grain_map, class_labels):
        """
        Args:
            target_grain_map: (B, 16, 16) - LongTensor with binary values {0, 1}
            class_labels: (B,) - LongTensor with ImageNet classes
        Returns:
            loss_dict: {'ce_loss': scalar}
        """
        device = target_grain_map.device
        batch_size = target_grain_map.shape[0]

        # 1. Prepare Inputs
        target_indices = target_grain_map.long() # Clean ground truth for Loss
        
        # Map binary {0, 1} to {-1.0, 1.0} for diffusion stability
        x_start = (target_grain_map.float() * 2.0) - 1.0
        x_start = x_start.unsqueeze(1) # (B, 1, 16, 16)

        # 2. Sample Timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

        # 3. Noise Injection (Forward Process)
        noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
        
        x_t = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

        # 4. Model Prediction
        # Predict logits for classification from the noisy map x_t
        # Output: (B, 2, 16, 16)
        logits = self._run_backbone(x_t, t, class_labels)

        # 5. Calculate Loss (Eq 5: Cross Entropy)
        # Target must be Long indices (0 or 1), Logits must be (B, C, H, W)
        loss_fn = nn.CrossEntropyLoss()
        ce_loss = loss_fn(logits, target_indices)

        return {
            "ce_loss": ce_loss
        }