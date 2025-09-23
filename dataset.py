import os
import torch
import zipfile
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms as T
from torch.utils.data import DataLoader
import logging


def pair(x):
    """Helper function to create tuple pairs"""
    return x, x


def get_transforms(resolution=256, is_train=True):
    """
    Create image transforms for training and validation
    
    Args:
        resolution: Target image resolution (default: 256)
        is_train: Whether this is for training (applies augmentations)
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    ops = []
    
    # Base transforms - resize and crop
    ops += [
        T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(resolution)
    ]
    
    # Training augmentations
    if is_train:
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            # Add more augmentations for better training
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.RandomRotation(degrees=5),
        ]
    
    # Convert to tensor and normalize to [-1, 1] range
    ops += [
        T.ToTensor(),  # [0, 1]
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # [-1, 1]
    ]
    
    return T.Compose(ops)


class CocoDataset(torch.utils.data.Dataset):
    """
    COCO Dataset for image reconstruction tasks
    """
    
    def __init__(self, root, dataType='train2017', annType='captions', 
                 is_train=True, resolution=256, max_examples=None):
        """
        Initialize COCO dataset
        
        Args:
            root: Path to COCO dataset root directory
            dataType: COCO data split ('train2017', 'val2017')
            annType: Annotation type ('captions', 'instances') 
            is_train: Whether this is training data (affects transforms)
            resolution: Target image resolution
            max_examples: Maximum number of examples to use (for debugging)
        """
        self.root = root
        self.dataType = dataType
        self.is_train = is_train
        self.resolution = resolution
        
        # Set up paths
        self.img_dir = os.path.join(root, dataType)
        annFile = os.path.join(root, 'annotations', f'{annType}_{dataType}.json')
        
        # Validate paths exist
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(annFile):
            raise FileNotFoundError(f"Annotation file not found: {annFile}")
        
        # Initialize COCO API
        self.coco = COCO(annFile)
        self.imgids = self.coco.getImgIds()
        
        # Limit dataset size if specified
        if max_examples and max_examples < len(self.imgids):
            self.imgids = self.imgids[:max_examples]
            logging.info(f"Limited dataset to {max_examples} examples")
        
        # Set up transforms
        self.transform = get_transforms(resolution=resolution, is_train=is_train)
        
        logging.info(f"Initialized COCO dataset: {len(self.imgids)} images from {dataType}")
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset
        
        Args:
            idx: Index of the item
            
        Returns:
            torch.Tensor: Transformed image tensor
        """
        # Handle potential index errors
        if idx >= len(self.imgids):
            idx = idx % len(self.imgids)
            
        imgid = self.imgids[idx]
        
        try:
            # Load image info and file
            img_info = self.coco.loadImgs(imgid)[0]
            img_name = img_info['file_name']
            img_path = os.path.join(self.img_dir, img_name)
            
            # Check if image file exists
            if not os.path.exists(img_path):
                logging.warning(f"Image not found: {img_path}, using next image")
                return self.__getitem__((idx + 1) % len(self.imgids))
            
            # Load and convert image
            img = Image.open(img_path).convert('RGB')
            
            # Check image validity
            if img.size[0] == 0 or img.size[1] == 0:
                logging.warning(f"Invalid image size: {img_path}, using next image")
                return self.__getitem__((idx + 1) % len(self.imgids))
            
            # Apply transforms
            if self.transform is not None:
                img = self.transform(img)
            
            return img
            
        except Exception as e:
            logging.warning(f"Error loading image {imgid}: {e}, using next image")
            return self.__getitem__((idx + 1) % len(self.imgids))
    
    def __len__(self):
        """Return dataset size"""
        return len(self.imgids)
    
    def get_image_info(self, idx):
        """Get metadata for an image (useful for debugging)"""
        imgid = self.imgids[idx]
        img_info = self.coco.loadImgs(imgid)[0]
        return img_info



def get_coco_loaders(root, batch_size=8, num_workers=4, resolution=256, 
                     max_train_examples=None, max_val_examples=None, 
                     pin_memory=True, include_captions=False):
    """
    Create COCO data loaders for training and validation
    
    Args:
        root: Path to COCO dataset root
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        resolution: Target image resolution
        max_train_examples: Limit training examples (for debugging)
        max_val_examples: Limit validation examples (for debugging)
        pin_memory: Whether to pin memory for faster GPU transfer
        include_captions: Whether to return captions along with images
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Choose dataset class
    DatasetClass =  CocoDataset
    
    # Create datasets
    train_ds = DatasetClass(
        root=root, 
        dataType='train2017', 
        annType='captions', 
        is_train=True,
        resolution=resolution,
        max_examples=max_train_examples
    )
    
    val_ds = DatasetClass(
        root=root, 
        dataType='val2017', 
        annType='captions', 
        is_train=False,
        resolution=resolution,
        max_examples=max_val_examples
    )
    
    # Create data loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  # Important for training!
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Ensures consistent batch sizes
        persistent_workers=num_workers > 0  # Keeps workers alive between epochs
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Keep all validation samples
        persistent_workers=num_workers > 0
    )
    
    logging.info(f"Created data loaders:")
    logging.info(f"  Train: {len(train_ds)} samples, {len(train_dl)} batches")
    logging.info(f"  Val: {len(val_ds)} samples, {len(val_dl)} batches")
    
    return train_dl, val_dl


# For backward compatibility with your existing code
def transform(resolution=256, is_train=True):
    """Backward compatibility wrapper"""
    return get_transforms(resolution, is_train)


class CoCo(CocoDataset):
    """Backward compatibility wrapper"""
    def __init__(self, root, dataType='train2017', annType='captions', is_train=True):
        super().__init__(root, dataType, annType, is_train)
        
    def __getitem__(self, idx):
        # Return both image and caption for backward compatibility
        img = super().__getitem__(idx)
        
        # Get caption
        imgid = self.imgids[idx]
        try:
            annid = self.coco.getAnnIds(imgIds=imgid)
            if len(annid) > 0:
                ann = np.random.choice(self.coco.loadAnns(annid))['caption']
            else:
                ann = "No caption available"
        except:
            ann = "Caption unavailable"
            
        return img, ann