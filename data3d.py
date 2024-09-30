from os import path, listdir
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm

class MyDataset(data.Dataset):
    def __init__(self, root_dirs, transform, load_color=False, depth=16, image_size=(128, 128)):
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

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_path = self.video_dirs[idx]
        # Get all image files, sorted in order
        image_files = sorted([file for file in listdir(video_path) 
                              if file.endswith(('tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp'))])
        
        # Ensure there are enough frames
        if len(image_files) < self.depth:
            raise ValueError(f"Video {video_path} has fewer than {self.depth} frames.")
        
        # Select the first `depth` frames
        selected_files = image_files[:self.depth]
        frames = []
        for file in selected_files:
            img_path = path.join(video_path, file)
            img = Image.open(img_path)
            if not self.load_color:
                img = img.convert('L')  # Convert to grayscale
            if self.transform:
                img = self.transform(img)  # Apply transformations
            frames.append(img)
        
        # Stack frames into a 3D tensor
        video_tensor = torch.stack(frames, dim=1)  # (C, D, H, W)
        return video_tensor

def get_data_loader(dir_list, batch_size=1, load_color=False, crop_size=None, test=True, depth=16):
    xfm = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor()
    ])
    
    return data.DataLoader(MyDataset(dir_list, xfm, load_color, depth=depth, image_size=(crop_size, crop_size)),
                           batch_size=batch_size,
                           drop_last=(not test),
                           shuffle=(not test))

def get_fit_loaders(trn_path_list=['data_gen/data16/train'],
                   val_path_list=['data_gen/data16/val'],
                   tst_path_list=['data_gen/data16/test'],
                   crop_size=128,
                   batch_size=[10, 1, 1],
                   load_color=False,
                   depth=16):

    if type(batch_size) is int:
        batch_size = [batch_size, 1, 1]

    dataloaders = {
        'train': get_data_loader(trn_path_list, 
                                 batch_size=batch_size[0], 
                                 load_color=load_color, 
                                 crop_size=crop_size, 
                                 test=False,
                                 depth=depth),
        'val':   get_data_loader(val_path_list, 
                                 batch_size=batch_size[1], 
                                 load_color=load_color, 
                                 crop_size=crop_size, 
                                 test=True,
                                 depth=depth),
        'test':  get_data_loader(tst_path_list, 
                                 batch_size=batch_size[2], 
                                 load_color=load_color, 
                                 crop_size=crop_size, 
                                 test=True,
                                 depth=depth)
    }
    return dataloaders
