import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss that handles mixed precision properly"""
    def __init__(self, layers=['relu1_1', 'relu2_1', 'relu3_1']):  # Reduced layers for efficiency
        super().__init__()
        # Load VGG and extract features
        vgg = models.vgg16(pretrained=True).features.eval().cuda()
        
        # Layer mapping
        layer_map = {
            'relu1_1': 2, 'relu2_1': 7, 'relu3_1': 12, 'relu4_1': 19
        }
        
        # Build feature extractors
        self.feature_extractors = nn.ModuleList()
        self.layer_names = layers
        
        for layer_name in layers:
            layer_idx = layer_map[layer_name]
            feature_extractor = nn.Sequential(*list(vgg.children())[:layer_idx+1])
            # Ensure all parameters are frozen and in float32
            for param in feature_extractor.parameters():
                param.requires_grad = False
            feature_extractor.eval()
            self.feature_extractors.append(feature_extractor)
    
    def forward(self, x, target):
        # Convert inputs to float32 to avoid mixed precision issues
        with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for this computation
            x = x.float()
            target = target.float()
            
            # Normalize to VGG input range [0,1]
            x = (x + 1.0) / 2.0
            target = (target + 1.0) / 2.0
            
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
            
            x = (x - mean) / std
            target = (target - mean) / std
            
            # Compute perceptual loss
            total_loss = 0.0
            for extractor in self.feature_extractors:
                x_feat = extractor(x)
                target_feat = extractor(target)
                total_loss += F.mse_loss(x_feat, target_feat)
            
            return total_loss / len(self.feature_extractors)


def simple_dvae_loss(model_output, target_images, perceptual_loss_fn, weights=None):
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
def create_simple_loss_function():
    """Create the simple loss function"""
    perceptual_loss_fn = PerceptualLoss()
    
    def loss_fn(model_output, target_images):
        return simple_dvae_loss(
            model_output, 
            target_images, 
            perceptual_loss_fn,
            weights={
                'l1': 0.7,      # Higher weight for edge preservation
                'l2': 0.3,      # Lower weight for stability
                'perceptual': 0.1  # Moderate weight for visual quality
            }
        )
    return loss_fn