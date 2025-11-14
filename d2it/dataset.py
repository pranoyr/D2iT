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
import pickle as pkl
import logging

import cv2
import numpy as np


from torchvision import datasets
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


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
    # ops += [
    #     T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
    #     T.CenterCrop(resolution)
    # ]

    ops += [
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR)
    ]
    
    # # Training augmentations
    # if is_train:
    #     ops += [
    #         T.RandomHorizontalFlip(p=0.5),
    #         # Add more augmentations for better training
    #         T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    #         T.RandomRotation(degrees=5),
    #     ]
    
    # Convert to tensor and normalize to [-1, 1] range
    ops += [
        T.ToTensor(),  # [0, 1]
        # imagenet
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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




class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Simplified ImageNet loader - only returns integer labels
        
        Args:
            root_dir: Path to train or val folder with synset subfolders
            csv_file: Path to LOC_train_solution.csv or LOC_val_solution.csv
            transform: Image transforms
        """
        self.root_dir = root_dir
        if transform is not None:
            self.transform = transform
            print("Using custom transforms")
        else:
            self.transform = get_transforms(resolution=256, is_train=is_train)

        self.is_train = is_train

        csv_file = os.path.join(root_dir,
                                'LOC_train_solution.csv' if is_train else 'LOC_val_solution.csv')
        
        # Load CSV
        self.data = pd.read_csv(csv_file) 
        self.data.columns = ["ImageId", "Label"]
        
        # Create synset to integer mapping (sorted alphabetically for consistency)
        all_synsets = sorted(set(self.data["Label"].str.split(" ").str[0]))
        self.synset_to_idx = {synset: idx for idx, synset in enumerate(all_synsets)}
        
        print(f"Loaded {len(self.data)} images with {len(all_synsets)} classes")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img_name = self.data.iloc[idx, 0] + ".JPEG"

        img_name = str(self.data.iloc[idx, 0]).strip() + ".JPEG"
        
        # Extract synset (first word before space)
        label_field = str(self.data.iloc[idx, 1]).strip()
        label_synset = label_field.split(" ")[0]
        
        # Convert to integer label
        label = self.synset_to_idx[label_synset]
        
        # Load image from synset subfolder
        is_train = "train" if self.is_train else "val"
        img_path = os.path.join(self.root_dir, "Data/CLS-LOC", is_train,  label_synset, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, label




def get_imagenet_loaders(root_dir, batch_size=8, num_workers=4, pin_memory=True, transform=None):
    """
    Create ImageNet data loaders for training and validation
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, val_loader)
    """


    DatasetClass =  ImageNetDataset 

    train_ds = DatasetClass(
        root_dir=root_dir,
        transform=transform,
        is_train=True
    )


    val_ds = DatasetClass(
        root_dir=root_dir,
        transform=transform,
        is_train=False
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size,  # Paper uses 256
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader


def get_STL_loaders(root_dir, batch_size=8, num_workers=4, pin_memory=True, transform=None):
    """
    Create ImageNet data loaders for training and validation
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, val_loader)
    """


    DatasetClass =  ImageNetDataset 
    
    if transform is not None:
        train_transform = transform
    else: 
        train_transform = get_transforms(resolution=224, is_train=True)


    # Load STL-10
    train_ds = datasets.STL10(
        root=root_dir,
        split='train',
        download=True,
        transform=train_transform
    )

    if transform is not None:
        test_transform = transform
    else: 
        test_transform = get_transforms(resolution=224, is_train=False)

    val_ds = datasets.STL10(
        root=root_dir,
        split='test',
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size,  # Paper uses 256
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, extensions=(".jpg", ".png", ".jpeg",".pkl")):
        """
        Args:
            root_dir (str): Path to the root image folder.
            transform (callable, optional): Transform to apply to each image.
            extensions (tuple): Allowed file extensions.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = extensions

        # Collect all image file paths recursively
        self.image_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(root_dir)
            for file in files
            if file.lower().endswith(extensions)
        ]

        # # number of samples  = 1
        # self.image_paths = self.image_paths[:100]

        if not self.image_paths:
            raise RuntimeError(f"No images found in {root_dir} with extensions {extensions}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # load grain map from pkl
        if path.lower().endswith(".pkl"):
            with open(path, 'rb') as f:
                pkl_file  = pkl.load(f) 
        grain_map = pkl_file['grain_map']
        label = pkl_file['label']
        labels = torch.tensor(label)
        
        # convert to long int
        return torch.tensor(grain_map).long(), label



def get_loaders(root_dir, batch_size=8, num_workers=4, pin_memory=True):
    """Wrapper to get ImageNet loaders for backward compatibility"""
    
    # dataset loader
    train_ds = ImageFolderDataset(
        root_dir=root_dir)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    return train_dl, None




# train_dl = get_loaders("./data/grain_maps", batch_size=1, num_workers=0)[0]
# for batch in train_dl:
#     grain_map, label = batch
#     print(grain_map.shape, label)



#     # permute and conver to numpy
#     # grain_map_np = grain_map.permute(1,2,0).numpy()
#     # print(grain_map_np.shape)
#     # exit()

#     grain_map_np = (grain_map.cpu().numpy() * 255).astype(np.uint8)
#     print(grain_map_np.shape)
#     cv2.imwrite("grain_map.jpg", grain_map_np[0])
#     # cv2.imshow("grain_map", grain_map_np)
#     # cv2.waitKey(0)
#     exit()

 




# # Create datasets
# train_dataset = ImageNetDataset(
#     root_dir="/media/pranoy/Datasets/ILSVRC",
# )




# for i  in range(len(train_dataset)):
#     img, label = train_dataset[i]
#     print(label)
    



# # Create dataloaders
# train_loader = DataLoader(
#     train_dataset, 
#     batch_size=1,  # Paper uses 256
#     shuffle=True,
#     num_workers=0,
#     pin_memory=True,
#     drop_last=True
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=1,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=True
# )

