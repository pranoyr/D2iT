import torch
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d
import argparse
from dataset import get_coco_loaders
from dvae import DVAE
from tqdm.auto import tqdm
import torch.nn.functional as F


from torchvision.models import inception_v3
import torch.nn as nn
import torch.nn.functional as F

class InceptionV3_FID(nn.Module):
    def __init__(self):
        super().__init__()
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.Mixed_7c.register_forward_hook(self.output_hook)  # hook at pool3 input
        inception.eval()
        for p in inception.parameters():
            p.requires_grad = False
        self.inception = inception
        self.output = None

    def output_hook(self, module, input, output):
        # After Mixed_7c, before final avgpool
        self.output = output

    def forward(self, x):
        # Resize to 299x299 and ensure 3 channels
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        _ = self.inception(x)  # forward pass
        feat = self.output
        feat = F.adaptive_avg_pool2d(feat, output_size=(1, 1))  # pool to 2048
        return feat.view(feat.size(0), -1)


# -----------------------------
# FID computation function
# -----------------------------
def compute_fid_score(model, real_loader, device="cuda", num_samples=10000):
    inception = InceptionV3_FID().to(device)

    real_feats, fake_feats = [], []
    seen = 0

    with torch.no_grad():
        for images in tqdm(real_loader, desc="Computing FID"):
            images = images.to(device)

            # Get reconstructions
            recons = model(images)["reconstructed"]
            
            # Extract features
            real_feats.append(inception(images).cpu().numpy())
            fake_feats.append(inception(recons).cpu().numpy())

            seen += images.size(0)
            if seen >= num_samples:
                break

    real_feats = np.concatenate(real_feats, axis=0)[:num_samples]
    fake_feats = np.concatenate(fake_feats, axis=0)[:num_samples]

    # Compute mean & covariance
    mu_real, sigma_real = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)

    # Fréchet distance
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)

    return float(fid)


# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/media/pranoy/Datasets/coco-dataset/coco',
                        help="Path to dataset")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--num_samples', type=int, default=10000, help="Number of samples for FID")

    args = parser.parse_args()

    # Load model
    model = DVAE().cuda()
    checkpoint = torch.load(args.ckpt, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare dataloader
    _, val_dl = get_coco_loaders(
        root=args.root,
        batch_size=32,
        resolution=256,
        num_workers=4,
        max_val_examples=args.num_samples
    )

    # Compute FID score
    fid = compute_fid_score(
        model=model,
        real_loader=val_dl,
        device='cuda',
        num_samples=args.num_samples
    )
    
    print(f"Final FID Score: {fid:.2f}")
