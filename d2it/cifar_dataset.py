from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transform for 256×256
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load STL-10
train_dataset = datasets.STL10(
    root='./data',
    split='train',
    download=True,
    transform=transform
)

test_dataset = datasets.STL10(
    root='./data',
    split='test',
    download=True,
    transform=transform
)

# # Optional: unlabeled data (100k images)
# unlabeled_dataset = datasets.STL10(
#     root='./data',
#     split='unlabeled',
#     download=True,
#     transform=transform
# )

# # Info
# print(f"✅ STL-10 Dataset Loaded")
# print(f"Classes: {len(train_dataset.classes)}")
# print(f"Class names: {train_dataset.classes}")
# print(f"\nTrain: {len(train_dataset)} images")
# print(f"Test: {len(test_dataset)} images")
# print(f"Unlabeled: {len(unlabeled_dataset)} images")

# # Check a sample
# image, label = train_dataset[0]
# print(f"\nSample:")
# print(f"Image shape: {image.shape}")
# print(f"Label (int): {label}")
# print(f"Label (name): {train_dataset.classes[label]}")

# # Create dataloaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# # Test batch
# for images, labels in train_loader:
#     print(f"\n✅ Batch:")
#     print(f"Images shape: {images.shape}")  # [32, 3, 256, 256]
#     print(f"Labels shape: {labels.shape}")  # [32]
#     print(f"Labels: {labels[:10]}")
#     print(f"Label names: {[train_dataset.classes[l.item()] for l in labels[:5]]}")
#     break