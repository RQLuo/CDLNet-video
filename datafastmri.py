import os
from os import path, listdir
import h5py
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import random
import fastmri
from fastmri.data import transforms as T
from PIL import Image

class FastMRIDataset(data.Dataset):
    def __init__(self, root_dirs, transform=None, load_color=False, depth=16, image_size=(128, 128), 
                 test=False, crop_ratio=0.5):
        """
        Initialize the FastMRI dataset.

        Args:
            root_dirs (list): List of root directories containing H5 files.
            transform (callable, optional): Transformations to apply to the images.
            load_color (bool): Whether to load color images (usually False).
            depth (int): Number of consecutive slices per sample.
            image_size (tuple): Size of the cropping window (width, height).
            test (bool): Whether the dataset is for testing.
            crop_ratio (float): Probability of applying cropping (only cropping is retained here, so this parameter is not used).
        """
        self.h5_files = []
        for cur_path in root_dirs:
            # Get all H5 files in the directory
            files = [path.join(cur_path, file) for file in listdir(cur_path) 
                     if file.lower().endswith('.h5')]
            self.h5_files += files
        
        print(f"Loading H5 files from {root_dirs}: {len(self.h5_files)} files found.")
        self.depth = depth
        self.load_color = load_color
        self.transform = transform
        self.image_size = image_size
        self.test = test
        self.crop_ratio = crop_ratio  # Only retain cropping
    
    def __len__(self):
        return len(self.h5_files)
    
    def __getitem__(self, idx):
        """
        Retrieve the data sample at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            torch.Tensor: Tensor of shape (C, D, H, W).
        """
        h5_file = self.h5_files[idx]
        with h5py.File(h5_file, 'r') as hf:
            volume_kspace = hf['kspace'][()]  # Shape: (num_slices, 640, 368)
            num_slices = volume_kspace.shape[0]
            
            if num_slices < self.depth:
                raise ValueError(f"H5 file {h5_file} has fewer slices ({num_slices}) than depth ({self.depth}).")
            
            # Randomly select a starting index such that start_idx + depth <= num_slices
            start_idx = random.randint(0, num_slices - self.depth)
            slice_kspace = volume_kspace[start_idx:start_idx + self.depth]  # Shape: (depth, 640, 368)
        
        frames = []
        crop_area = None  # Ensure the same cropping area is applied to all frames
    
        for i in range(self.depth):
            current_kspace = slice_kspace[i]  # Shape: (640, 368)
            slice_kspace_tensor = T.to_tensor(current_kspace)  # Convert to PyTorch tensor
            slice_image = fastmri.ifft2c(slice_kspace_tensor)  # Apply inverse Fourier transform
            slice_image_abs = fastmri.complex_abs(slice_image)  # Get the magnitude to obtain a real image
    
            # Convert to PIL image for cropping
            # Normalize the image to [0, 255] and convert to uint8
            slice_image_np = slice_image_abs.cpu().numpy()
            slice_image_np = (slice_image_np - slice_image_np.min()) / (slice_image_np.max() - slice_image_np.min())
            slice_image_np = (slice_image_np * 255).astype(np.uint8)
            img = Image.fromarray(slice_image_np)
    
            if not self.test:
                if crop_area is None:
                    # Randomly determine the cropping area based on the first image
                    img_width, img_height = img.size
                    crop_width, crop_height = self.image_size
                    if crop_width > img_width or crop_height > img_height:
                        raise ValueError(f"Crop size {self.image_size} is larger than image size {(img_width, img_height)}.")
                    
                    crop_x = random.randint(0, img_width - crop_width)
                    crop_y = random.randint(0, img_height - crop_height)
                    crop_area = (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
                
                # Apply the same cropping area to all frames
                img = img.crop(crop_area)
    
            if self.transform:
                img = self.transform(img)  # Apply transformations
            else:
                # Default transformation: convert to tensor
                img = transforms.ToTensor()(img)
    
            frames.append(img)
        
        # Stack frames into a 4D tensor (C, D, H, W)
        video_tensor = torch.stack(frames, dim=1)  # Shape: (C, D, H, W)
        return video_tensor


def get_fastmri_data_loader(dir_list, batch_size=1, load_color=False, crop_size=128, test=True, depth=16):
    """
    Create a DataLoader for the FastMRI dataset.

    Args:
        dir_list (list): List of directories containing H5 files.
        batch_size (int): Batch size.
        load_color (bool): Whether to load color images.
        crop_size (int): Size to crop the images.
        test (bool): Whether the DataLoader is for testing.
        depth (int): Number of consecutive slices per sample.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    if test:
        # During testing, only convert images to tensors without any augmentation
        xfm = transforms.ToTensor()
    else:
        # During training/validation, only perform conversion (cropping is handled in the dataset)
        xfm = transforms.ToTensor()
    
    return data.DataLoader(
        FastMRIDataset(
            root_dirs=dir_list, 
            transform=xfm, 
            load_color=load_color, 
            depth=depth, 
            image_size=(crop_size, crop_size), 
            test=test,
            crop_ratio=0.5  # Although only cropping is performed here
        ),
        batch_size=batch_size,
        drop_last=(not test),
        shuffle=(not test)
    )


def get_fit_loaders(trn_path_list=['data_gen/data16/train'],
                   val_path_list=['data_gen/data16/val'],
                   tst_path_list=['data_gen/data16/test'],
                   crop_size=128,
                   batch_size=[10, 1, 1],
                   load_color=False,
                   depth=16):
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        trn_path_list (list): List of training directories.
        val_path_list (list): List of validation directories.
        tst_path_list (list): List of testing directories.
        crop_size (int): Size to crop the images.
        batch_size (list or int): Batch sizes for training, validation, and testing.
        load_color (bool): Whether to load color images.
        depth (int): Number of consecutive slices per sample.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    if isinstance(batch_size, int):
        batch_size = [batch_size, 1, 1]
    
    dataloaders = {
        'train': get_fastmri_data_loader(
            trn_path_list, 
            batch_size=batch_size[0], 
            load_color=load_color, 
            crop_size=crop_size, 
            test=False,
            depth=depth
        ),
        'val': get_fastmri_data_loader(
            val_path_list, 
            batch_size=batch_size[1], 
            load_color=load_color, 
            crop_size=crop_size, 
            test=True,  # Validation typically uses the test setting
            depth=depth
        ),
        'test': get_fastmri_data_loader(
            tst_path_list, 
            batch_size=batch_size[2], 
            load_color=load_color, 
            crop_size=crop_size, 
            test=True,
            depth=depth
        )
    }
    return dataloaders
