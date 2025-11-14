# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from d2it.loss import ce_loss, MI_loss


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        # use_dropout = self.dropout_prob > 0
        # if (train and use_dropout) or (force_drop_ids is not None):
        #     labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ------------------ Q Network ------------------
class QNetwork(nn.Module):
    """Predicts noise vector ẑ from hidden transformer features"""
    def __init__(self, hidden_size, noise_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, noise_dim)
        )

    def forward(self, x):
        # x: [B, N, hidden_size]
        # Global average pooling across tokens
        # x_mean = x.mean(dim=1)  # [B, hidden_size]
        z_hat = self.net(x)  # [B, noise_dim]
        return z_hat




# class FinalLayer(nn.Module):
#     """
#     The final layer of DiT.
#     """
#     def __init__(self, hidden_size, granularity=2):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.linear = nn.Linear(hidden_size, granularity, bias=True)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 2 * hidden_size, bias=True)
#         )

#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
#         x = modulate(self.norm_final(x), shift, scale)
#         x = self.linear(x)
#         return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, granularity=2):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, granularity, bias=True)
    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x



class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        output_size=16,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        num_tokens = output_size * output_size
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size

        # IMPORTANT: Use learnable spatial tokens instead of random noise
        self.spatial_tokens = nn.Parameter(torch.zeros(1, num_tokens, hidden_size))
        
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, granularity=2)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_tokens ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # IMPORTANT: Initialize learnable spatial tokens with small values
        nn.init.trunc_normal_(self.spatial_tokens, std=0.02)

        # Initialize label embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, c, target=None):
        B = c.shape[0]
        
        # FIXED: Use learnable tokens, optionally with small noise for regularization
        x = self.spatial_tokens.expand(B, -1, -1)  # [B, num_tokens, hidden_size]
        
        # Optional: Add small noise during training for regularization (much smaller than before!)
        if self.training:
            x = x + torch.randn_like(x) * 0.01  # Very small noise, not completely random!
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Get class embedding
        c = self.y_embedder(c, self.training)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Predict grain map
        grain_map = self.final_layer(x)

        # Compute loss if target is provided
        if target is not None:
            loss = ce_loss(grain_map, target)
            loss_dict = {"ce_loss": loss}
            return loss_dict
        else:
            return grain_map




#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

    
def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)




if __name__ == "__main__":
    model = DiT_S_2()

    # number of params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_params/1e6:.2f}M") # Number of params: 23.56M

    c = torch.randint(0, 1000, (2,))
    x = model(c)
    print(x.shape) # torch.Size([2, 256, 2])


    x = rearrange(x, 'b n c -> (b n) c')
    gt = torch.randint(0, 2, (2, 16, 16)) # (b, h, w)


    gt = rearrange(gt, 'b h w -> (b h w)')


    # cross entropy loss
    loss = nn.CrossEntropyLoss()
    l = loss(x, gt)
    print(l) # tensor(2.3030, grad_fn=<NllLossBackward


    print(x.shape, gt.shape) # torch.Size([512, 4]) torch.Size([512, 4])

    
   
