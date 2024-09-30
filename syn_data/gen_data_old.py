import os
import shutil
from gen import gen_data
from PIL import Image
import numpy as np

# This script generates a specified number of videos with a defined number of frames.
# It creates subdirectories for saving frame images and generates animations from the data.

num_videos = 5  # Modify as needed
x, y, num_frames = 128, 128, 64  # 64 frames per video
data_gen_dir = 'data_gen'
data64_main = os.path.join(data_gen_dir, 'data64')
data32_main = os.path.join(data_gen_dir, 'data32')
data16_main = os.path.join(data_gen_dir, 'data16')

# Ensure the main data_gen directory and its subdirectories exist
os.makedirs(data64_main, exist_ok=True)
os.makedirs(data32_main, exist_ok=True)
os.makedirs(data16_main, exist_ok=True)

def create_subdirs(base_dir, video_idx):
    subdir = os.path.join(base_dir, str(video_idx))
    os.makedirs(subdir, exist_ok=True)
    return subdir

def sample_frame_indices(total, sample_size):
    step = total / sample_size
    return [int(step * i) for i in range(sample_size)]

for video_idx in range(1, num_videos + 1):
    print(f"Generating video {video_idx}/{num_videos}...")

    data64_dir = create_subdirs(data64_main, video_idx)
    data32_dir = create_subdirs(data32_main, video_idx)
    data16_dir = create_subdirs(data16_main, video_idx)

    data = gen_data(x, y, num_frames)  # Assuming this returns a numpy array of shape (x, y, num_frames)

    # Normalize data to 0-255 for image representation
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min == 0:
        normalized_data = np.zeros_like(data, dtype=np.uint8)
    else:
        normalized_data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)

    for frame in range(num_frames):
        frame_data = normalized_data[:, :, frame]
        frame_image = Image.fromarray(frame_data, mode='L')  # 'L' mode for (8-bit pixels, black and white)
        frame_filename = os.path.join(data64_dir, f'frame_{frame:03d}.png')
        frame_image.save(frame_filename)

    sampled32 = sample_frame_indices(num_frames, 32)
    sampled16 = sample_frame_indices(num_frames, 16)

    def copy_samples(sampled, src_dir, dst_dir):
        for frame in sampled:
            src = os.path.join(src_dir, f'frame_{frame:03d}.png')
            dst = os.path.join(dst_dir, f'frame_{frame:03d}.png')
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"Warning: {src} does not exist, unable to copy.")

    copy_samples(sampled32, data64_dir, data32_dir)
    copy_samples(sampled16, data64_dir, data16_dir)

    print(f"Video {video_idx} generated successfully.\n")

print("All videos and their frames have been successfully generated and saved.")
