import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Callable


class MRIDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "Training",
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        num_classes: int = 4
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.num_classes = num_classes
        
        self.class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        self.samples = self._load_samples()
        self.class_weights = self._compute_class_weights()
        
    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        split_dir = os.path.join(self.data_dir, self.split)
        
        for class_name in self.class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(class_dir, filename)
                    samples.append((filepath, class_idx))
        
        return samples
    
    def _compute_class_weights(self) -> torch.Tensor:
        class_counts = torch.zeros(self.num_classes)
        for _, label in self.samples:
            class_counts[label] += 1
        
        total_samples = len(self.samples)
        weights = total_samples / (self.num_classes * class_counts)
        return weights
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]
        
        image = Image.open(filepath).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = self.transform(image)
        
        return image, label


class MRISliceDataset(Dataset):
    def __init__(
        self,
        volume_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        num_slices: int = 16,
        slice_selection: str = 'uniform'
    ):
        self.volume_paths = volume_paths
        self.labels = labels
        self.transform = transform
        self.num_slices = num_slices
        self.slice_selection = slice_selection
        
    def _select_slices(self, volume: np.ndarray) -> np.ndarray:
        depth = volume.shape[2]
        
        if self.slice_selection == 'uniform':
            indices = np.linspace(0, depth-1, self.num_slices).astype(int)
        elif self.slice_selection == 'center':
            start = max(0, depth//2 - self.num_slices//2)
            end = min(depth, start + self.num_slices)
            indices = np.arange(start, end)
        else:
            indices = np.random.choice(depth, self.num_slices, replace=False)
            indices = np.sort(indices)
        
        return volume[:, :, indices]
    
    def __len__(self) -> int:
        return len(self.volume_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        volume_path = self.volume_paths[idx]
        label = self.labels[idx]
        
        volume = np.load(volume_path)
        slices = self._select_slices(volume)
        
        if self.transform:
            slice_list = []
            for i in range(slices.shape[2]):
                slice_2d = slices[:, :, i]
                if isinstance(self.transform, A.Compose):
                    transformed = self.transform(image=slice_2d)
                    slice_list.append(transformed['image'])
                else:
                    slice_list.append(self.transform(slice_2d))
            
            slices = torch.stack(slice_list, dim=0)
        
        return slices, label


def get_transforms(
    split: str = "train",
    image_size: Tuple[int, int] = (224, 224),
    use_albumentations: bool = True
) -> Callable:
    
    if use_albumentations:
        if split == "train":
            transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ElasticTransform(p=0.3, alpha=1, sigma=50, alpha_affine=50),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=3),
                    A.MotionBlur(blur_limit=3),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.RandomGamma(gamma_limit=(0.8, 1.2)),
                    A.CLAHE(clip_limit=2.0),
                ], p=0.5),
                A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    else:
        if split == "train":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    return transform
def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    use_weighted_sampling: bool = True
) -> Dict[str, DataLoader]:
    
    train_transform = get_transforms("train", image_size)
    val_transform = get_transforms("val", image_size)
    
    train_dataset = MRIDataset(
        data_dir=data_dir,
        split="Training",
        transform=train_transform,
        target_size=image_size
    )
    
    val_dataset = MRIDataset(
        data_dir=data_dir,
        split="Testing",
        transform=val_transform,
        target_size=image_size
    )
    
    if use_weighted_sampling:
        weights = train_dataset.class_weights
        sample_weights = [weights[label] for _, label in train_dataset.samples]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset
    }