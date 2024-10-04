from os import path, listdir
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
import random

class MyDataset(data.Dataset):
    def __init__(self, root_dirs, transform, load_color=False, depth=16, image_size=(128, 128), test=False):
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

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_path = self.video_dirs[idx]
        # Get all image files, sorted in order
        image_files = sorted([file for file in listdir(video_path) 
                              if file.lower().endswith(('ti f', 'tiff', 'png', 'jpg', 'jpeg', 'bmp'))])
        
        # Ensure there are enough frames
        num_frames = len(image_files)
        if num_frames < self.depth:
            raise ValueError(f"Video {video_path} has fewer than {self.depth} frames.")
        
        # Randomly select a starting frame index such that we can pick `depth` consecutive frames
        start_idx = random.randint(0, num_frames - self.depth)
        selected_files = image_files[start_idx:start_idx + self.depth]

        if not self.test:
            # With 0.5 probability, reverse the selected frames (i.e., reverse playback)
            if random.random() < 0.5:
                selected_files = selected_files[::-1]  # Reverse the list of frames

        frames = []
        crop_area = None  # Placeholder for cropping area
        if not self.test:
            apply_crop = random.random() < 1  # Decide whether to apply cropping or resizing

        for file in selected_files:
            img_path = path.join(video_path, file)
            img = Image.open(img_path)
            if not self.load_color:
                img = img.convert('L')  # Convert to grayscale
            
            if not self.test:
                # Randomly decide to resize or crop, but ensure consistent operation across frames
                if apply_crop:
                    if crop_area is None:
                        # If crop area has not been set, randomly determine it
                        img_width, img_height = img.size
                        crop_x = random.randint(0, img_width - self.image_size[0])
                        crop_y = random.randint(0, img_height - self.image_size[1])
                        crop_area = (crop_x, crop_y, crop_x + self.image_size[0], crop_y + self.image_size[1])
                    #if random.random() < 0.2:
                    #    crop_x = random.randint(0, img_width - self.image_size[0])
                    #    crop_y = random.randint(0, img_height - self.image_size[1])
                    #    crop_area = (crop_x, crop_y, crop_x + self.image_size[0], crop_y + self.image_size[1])
                    img = img.crop(crop_area)  # Apply the same crop area to all frames

            if self.transform:
                img = self.transform(img)  # Apply transformations
            frames.append(img)
        
        # Stack frames into a 3D tensor
        video_tensor = torch.stack(frames, dim=1)  # (C, D, H, W)
        return video_tensor

def get_data_loader(dir_list, batch_size=1, load_color=False, crop_size=None, test=True, depth=16):
    if test:
        # During testing, only convert images to tensors without any augmentation
        xfm = transforms.ToTensor()
    else:
        # During training/validation, apply resizing and other transformations
        xfm = transforms.Compose([
            #transforms.RandomCrop(crop_size),
            transforms.ToTensor()
        ])
    
    return data.DataLoader(
        MyDataset(
            root_dirs=dir_list, 
            transform=xfm, 
            load_color=load_color, 
            depth=depth, 
            image_size=(crop_size, crop_size), 
            test=test
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
    
    if isinstance(batch_size, int):
        batch_size = [batch_size, 1, 1]
    
    dataloaders = {
        'train': get_data_loader(
            trn_path_list, 
            batch_size=batch_size[0], 
            load_color=load_color, 
            crop_size=crop_size, 
            test=False,
            depth=depth
        ),
        'val': get_data_loader(
            val_path_list, 
            batch_size=batch_size[1], 
            load_color=load_color, 
            crop_size=crop_size, 
            test=True,  # Typically validation uses test-like settings
            depth=depth
        ),
        'test': get_data_loader(
            tst_path_list, 
            batch_size=batch_size[2], 
            load_color=load_color, 
            crop_size=crop_size, 
            test=True,
            depth=depth
        )
    }
    return dataloaders
