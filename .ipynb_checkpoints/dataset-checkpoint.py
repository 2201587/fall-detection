import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from collections import Counter

# Define dataset class
class FallDetectionDataset(Dataset):
    def __init__(self, split_file, split, normalise):
        self.split = split
        self.normalise = normalise
        
        # Load split information
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        self.clips = split_data[split]['clips']
        
        # Verify resolution info exists
        if self.clips and 'width' not in self.clips[0]:
            raise ValueError("Resolution information not found in metadata.")
        
        print(f"\nLoaded {split} set:")
        print(f"  Total clips: {len(self.clips)}")
        print(f"  Fall clips: {split_data[split]['num_fall']}")
        print(f"  No-fall clips: {split_data[split]['num_no_fall']}")
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        
        # Load keypoints
        keypoints = np.load(clip_info['clip_file'])
        
        if self.normalise:
            # Normalise using this specific video's resolution
            width = clip_info['width']
            height = clip_info['height']
            keypoints = self.normalise_keypoints(keypoints, width, height)
        
        # Convert to tensor
        keypoints_tensor = torch.from_numpy(keypoints).float()
        
        # Get label
        label = clip_info['label']  # 0 = no fall, 1 = fall
        
        return {
            'keypoints': keypoints_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'video_name': clip_info['original_video'],
            'clip_index': clip_info['clip_index'],
            'resolution': f"{clip_info['width']}x{clip_info['height']}"
        }
    
    def normalise_keypoints(self, keypoints, width, height):
        normalised = keypoints.copy()
        
        # Normalise x coordinates (column 0) by actual width
        normalised[:, :, 0] = normalised[:, :, 0] / width
        
        # Normalise y coordinates (column 1) by actual height
        normalised[:, :, 1] = normalised[:, :, 1] / height
        
        # Keep confidence scores (column 2) as it is
        
        # Clip to [0, 1] range (in case any coordinates are outside frame)
        normalised[:, :, 0] = np.clip(normalised[:, :, 0], 0, 1)
        normalised[:, :, 1] = np.clip(normalised[:, :, 1], 0, 1)
        
        return normalised

def create_dataloaders(split_file, batch_size, num_workers=4, normalise=True):
    # Create datasets
    train_dataset = FallDetectionDataset(split_file, split='train', normalise=normalise)
    val_dataset = FallDetectionDataset(split_file, split='val', normalise=normalise)
    test_dataset = FallDetectionDataset(split_file, split='test', normalise=normalise)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True  # Drop incomplete last batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader