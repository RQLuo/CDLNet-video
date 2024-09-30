import os
import shutil
from gen import gen_data
from PIL import Image
import numpy as np

# This script generates a specified number of videos with a defined number of frames.
# It creates subdirectories for saving frame images.

num_videos = 2000  # Modify as needed
x, y, total_frames = 128, 128, 64  # Generate 64 frames per video
num_frames_to_save = 16  # Save only the first 16 frames
data_gen_dir = 'data_gen'
data16_main = os.path.join(data_gen_dir, 'data16')

# Ensure the main data_gen directory and its subdirectories exist
os.makedirs(data16_main, exist_ok=True)

def create_subdirs(base_dir, video_idx):
    subdir = os.path.join(base_dir, str(video_idx))
    os.makedirs(subdir, exist_ok=True)
    return subdir

for video_idx in range(1, num_videos + 1):
    print(f"Generating video {video_idx}/{num_videos}...")

    data16_dir = create_subdirs(data16_main, video_idx)

    # Generate data with the specified number of frames
    data = gen_data(x, y, total_frames)  # Assuming this returns a numpy array of shape (x, y, total_frames)

    # Normalize data to 0-255 for image representation
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min == 0:
        normalized_data = np.zeros_like(data, dtype=np.uint8)
    else:
        normalized_data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)

    # Save only the first 16 frames
    for frame in range(num_frames_to_save):
        frame_data = normalized_data[:, :, frame]
        frame_image = Image.fromarray(frame_data, mode='L')
        frame_filename = os.path.join(data16_dir, f'frame_{frame:03d}.png')
        frame_image.save(frame_filename)

print("All videos and their frames have been successfully generated and saved.")
