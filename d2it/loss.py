import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lpips
from einops import rearrange



class LPIPSPerceptualLoss(nn.Module):
    """LPIPS-based perceptual loss"""
    def __init__(self, net='vgg'):
        super().__init__()
        # Initialize LPIPS loss
        self.lpips_fn = lpips.LPIPS(net=net).eval()
        
        # Freeze all parameters
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def forward(self, x, target):
        # LPIPS expects inputs in [-1, 1] range, which matches your data
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            target = target.float()
            
            # LPIPS returns per-sample loss, average across batch
            loss = self.lpips_fn(x, target)
            return loss.mean()


def dvae_loss(model_output, target_images, perceptual_loss_fn, weights=None):
    """
    Simple DVAE loss: L1 + L2 + Perceptual
    
    Args:
        model_output: Dict with 'reconstructed' key
        target_images: Original input images [B, 3, H, W]
        perceptual_loss_fn: Perceptual loss function
        weights: Dict with loss component weights
    """
    if weights is None:
        weights = {
            'l1': 0.7,
            'l2': 0.3, 
            'perceptual': 0.1
        }
    
    reconstructed = model_output['reconstructed']
    
    # L1 Loss (preserves edges)
    l1_loss = F.l1_loss(reconstructed, target_images)
    
    # L2 Loss (smooth gradients, stable training)
    l2_loss = F.mse_loss(reconstructed, target_images)
    
    # Perceptual Loss (visual quality)
    perceptual_loss = perceptual_loss_fn(reconstructed, target_images)
    
    # Combined loss
    total_loss = (
        weights['l1'] * l1_loss + 
        weights['l2'] * l2_loss + 
        weights['perceptual'] * perceptual_loss
    )
    
    loss_dict = {
        'l1_loss': l1_loss.item(),
        'l2_loss': l2_loss.item(),
        'perceptual_loss': perceptual_loss.item(),
        'total_loss': total_loss.item()
    }
    
    return total_loss, loss_dict


# Usage in your training code:
def create_dvae_loss():
    """Create the simple loss function"""
    perceptual_loss_fn = LPIPSPerceptualLoss(net='vgg').cuda()
    
    def loss_fn(model_output, target_images, weights=None):
        return dvae_loss(
            model_output, 
            target_images, 
            perceptual_loss_fn,
            weights=weights
        )
    return loss_fn

def ce_loss(logits, targets):

    # print("logits shape:", logits.shape)
    # print("targets shape:", targets.shape)
    # exit()


    # Flatten logits and targets
    targets = rearrange(targets, 'b h w -> (b h w)')
    logits = rearrange(logits, 'b n c -> (b n) c')

 

    # targets = rearrange(targets, 'b h w -> b (h w)')
    # logits = rearrange(logits, 'b n c -> b c n')




    """Cross-entropy loss for classification tasks"""
    return F.cross_entropy(logits, targets)



def MI_loss(z, z_hat):
    """L_MI = E[||z - ẑ||²]"""
    return F.mse_loss(z_hat, z)