from os import path, listdir
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
import random

class MyDataset(data.Dataset):
    def __init__(self, root_dirs, transform, load_color=False, depth=16, image_size=(128, 128), 
                 test=False, crop_ratio=0.5, aug_prob=0.3, max_shift=10):
        """
        Initialize the dataset.

        Args:
            root_dirs (list): List of root directories containing video folders.
            transform (callable): Transformation to apply to the images.
            load_color (bool): Whether to load images in color.
            depth (int): Number of frames to sample.
            image_size (tuple): Size of the cropping window (width, height).
            test (bool): Whether the dataset is for testing.
            crop_ratio (float): Probability of applying cropping vs resizing.
            aug_prob (float): Probability of applying the new augmentation.
            max_shift (int): Maximum pixels to shift the cropping window during random walk.
        """
        self.video_dirs = []
        for cur_path in root_dirs:
            # Get all subfolders (each subfolder corresponds to a video)
            video_folders = [path.join(cur_path, folder) for folder in listdir(cur_path) 
                             if path.isdir(path.join(cur_path, folder))]
            self.video_dirs += video_folders
        
        print(f"Loading videos from {root_dirs}:")
        self.depth = depth
        self.load_color = load_color
        self.transform = transform
        self.image_size = image_size
        self.test = test
        self.crop_ratio = crop_ratio  # Probability to apply cropping
        self.aug_prob = aug_prob      # Probability to apply the new augmentation
        self.max_shift = max_shift    # Maximum shift for random walk

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_path = self.video_dirs[idx]
        # Get all image files, sorted in order
        image_files = sorted([file for file in listdir(video_path) 
                              if file.lower().endswith(('tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp'))])
        
        # Ensure there are enough frames
        num_frames = len(image_files)
        if num_frames < self.depth:
            raise ValueError(f"Video {video_path} has fewer than {self.depth} frames.")
        
        frames = []
        
        if not self.test and random.random() < self.aug_prob:
            # Apply the new augmentation: random walk cropping
            # Select a random starting frame
            start_frame_idx = random.randint(0, num_frames - 1)
            selected_files = image_files[start_frame_idx:start_frame_idx + self.depth]
            if len(selected_files) < self.depth:
                # If not enough frames from start_frame_idx, wrap around
                selected_files += image_files[:self.depth - len(selected_files)]
            
            # Initialize crop position
            first_img_path = path.join(video_path, selected_files[0])
            first_img = Image.open(first_img_path)
            if not self.load_color:
                first_img = first_img.convert('L')  # Convert to grayscale
            img_width, img_height = first_img.size

            crop_width, crop_height = self.image_size
            # Ensure the crop size is smaller than the image size
            if crop_width > img_width or crop_height > img_height:
                raise ValueError(f"Crop size {self.image_size} is larger than image size {(img_width, img_height)}.")
            
            # Randomly select initial crop position
            current_x = random.randint(0, img_width - crop_width)
            current_y = random.randint(0, img_height - crop_height)
            crop_area = (current_x, current_y, current_x + crop_width, current_y + crop_height)
            
            for file in selected_files:
                img_path = path.join(video_path, file)
                img = Image.open(img_path)
                if not self.load_color:
                    img = img.convert('L')  # Convert to grayscale
                
                # Perform random walk for crop position
                shift_x = random.randint(-self.max_shift, self.max_shift)
                shift_y = random.randint(-self.max_shift, self.max_shift)
                new_x = min(max(current_x + shift_x, 0), img_width - crop_width)
                new_y = min(max(current_y + shift_y, 0), img_height - crop_height)
                crop_area = (new_x, new_y, new_x + crop_width, new_y + crop_height)
                current_x, current_y = new_x, new_y  # Update current position
                
                # Crop the image
                img = img.crop(crop_area)
                
                if self.transform:
                    img = self.transform(img)  # Apply transformations
                frames.append(img)
        else:
            # Apply existing frame selection and augmentation
            # Randomly select a starting frame index such that we can pick `depth` consecutive frames
            start_idx = random.randint(0, num_frames - self.depth)
            selected_files = image_files[start_idx:start_idx + self.depth]

            if not self.test:
                if random.random() < 0.5:
                    selected_files = selected_files[::-1]  # Reverse the list of frames

            crop_area = None  # Placeholder for cropping area
            if not self.test:
                apply_crop = random.random() < self.crop_ratio  # Decide whether to apply cropping or resizing

            for file in selected_files:
                img_path = path.join(video_path, file)
                img = Image.open(img_path)
                if not self.load_color:
                    img = img.convert('L')  # Convert to grayscale
                
                if not self.test:
                    if apply_crop:
                        if crop_area is None:
                            # If crop area has not been set, randomly determine it
                            img_width, img_height = img.size
                            crop_x = random.randint(0, img_width - self.image_size[0])
                            crop_y = random.randint(0, img_height - self.image_size[1])
                            crop_area = (crop_x, crop_y, crop_x + self.image_size[0], crop_y + self.image_size[1])
                        img = img.crop(crop_area)  # Apply the same crop area to all frames

                if self.transform:
                    img = self.transform(img)  # Apply transformations
                frames.append(img)
        
        # Stack frames into a 4D tensor (C, D, H, W)
        video_tensor = torch.stack(frames, dim=1)  # (C, D, H, W)
        return video_tensor

def get_data_loader(dir_list, batch_size=1, load_color=False, crop_size=None, test=True, depth=16, 
                   crop_ratio=0.5, aug_prob=0.3, max_shift=10):
    """
    Create a DataLoader for the dataset.

    Args:
        dir_list (list): List of directories containing video data.
        batch_size (int): Batch size.
        load_color (bool): Whether to load images in color.
        crop_size (int): Size for cropping the images.
        test (bool): Whether the DataLoader is for testing.
        depth (int): Number of frames per sample.
        crop_ratio (float): Probability of applying cropping vs resizing.
        aug_prob (float): Probability of applying the new augmentation.
        max_shift (int): Maximum pixels to shift the cropping window during random walk.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    if test:
        # During testing, only convert images to tensors without any augmentation
        xfm = transforms.ToTensor()
    else:
        # During training/validation, apply resizing and other transformations
        xfm = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()
        ])
    
    return data.DataLoader(
        MyDataset(
            root_dirs=dir_list, 
            transform=xfm, 
            load_color=load_color, 
            depth=depth, 
            image_size=(crop_size, crop_size) if crop_size else (128, 128), 
            test=test,
            crop_ratio=crop_ratio,
            aug_prob=aug_prob,
            max_shift=max_shift
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
                   depth=16,
                   crop_ratio=0.5,
                   aug_prob=0.3,
                   max_shift=10):
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        trn_path_list (list): List of training directories.
        val_path_list (list): List of validation directories.
        tst_path_list (list): List of testing directories.
        crop_size (int): Size for cropping the images.
        batch_size (list or int): Batch sizes for train, val, and test.
        load_color (bool): Whether to load images in color.
        depth (int): Number of frames per sample.
        crop_ratio (float): Probability of applying cropping vs resizing.
        aug_prob (float): Probability of applying the new augmentation.
        max_shift (int): Maximum pixels to shift the cropping window during random walk.

    Returns:
        dict: Dictionary containing DataLoaders for 'train', 'val', and 'test'.
    """
    if isinstance(batch_size, int):
        batch_size = [batch_size, 1, 1]
    
    dataloaders = {
        'train': get_data_loader(
            trn_path_list, 
            batch_size=batch_size[0], 
            load_color=load_color, 
            crop_size=crop_size, 
            test=False,
            depth=depth,
            crop_ratio=crop_ratio,
            aug_prob=aug_prob,
            max_shift=max_shift
        ),
        'val': get_data_loader(
            val_path_list, 
            batch_size=batch_size[1], 
            load_color=load_color, 
            crop_size=crop_size, 
            test=True,  # Typically validation uses test-like settings
            depth=depth,
            crop_ratio=crop_ratio,
            aug_prob=aug_prob,
            max_shift=max_shift
        ),
        'test': get_data_loader(
            tst_path_list, 
            batch_size=batch_size[2], 
            load_color=load_color, 
            crop_size=crop_size, 
            test=True,
            depth=depth,
            crop_ratio=crop_ratio,
            aug_prob=aug_prob,
            max_shift=max_shift
        )
    }
    return dataloaders
