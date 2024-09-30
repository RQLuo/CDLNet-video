'''
This code generates a 3D animated visualization to show
slices of the 3D image moving through the third dimension.
'''
from gen import gen_data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

x, y, num_frames = 128, 128, 64
fig, axes = plt.subplots(3, 3, figsize=(9, 9))

ims = []
data_list = []

for i in range(3):
    for j in range(3):
        data = gen_data(x, y, num_frames)
        data_list.append(data)
        im = axes[i, j].imshow(data[:, :, 0], cmap='gray', animated=True)
        ims.append(im)
        axes[i, j].axis('off')

def update(frame):
    for idx, im in enumerate(ims):
        im.set_array(data_list[idx][:, :, frame]) 
    return ims
ani = FuncAnimation(fig, update, frames=range(num_frames), interval=15, blit=True)
gif_name = f"animation.gif"
writer = PillowWriter(fps=10)
ani.save(gif_name, writer=writer)
plt.close()
