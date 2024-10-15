import os
import sys
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datafastmri import get_fit_loaders
from model.net import CDLNet_CSR
from utils import async_prefetch_to_gpu
from utils import awgn

def main(args):
    """Load data, initialize the model, and train it based on the parameter dictionary."""
    ngpu = torch.cuda.device_count()  # Get the number of available GPUs
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")  # Use the first GPU if available
    print(f"Number of available GPUs: {ngpu}")
    print(f"Using device: {device}")

    model_args, train_args, paths = [args[item] for item in ['model', 'train', 'paths']]
    loaders = get_fit_loaders(**train_args['loaders'])  # Load the data
    net, opt, sched, epoch0 = init_model(args, device=device)  # Initialize model and optimizer

    fit(net, 
        opt, 
        loaders,
        sched=sched,
        save_dir=paths['save'],
        start_epoch=epoch0 + 1,
        device=device,
        **train_args['fit'],
        epoch_fun=lambda epoch_num: save_args(args, epoch_num))  # Train the model


def fit(net, opt, loaders,
        sched=None,
        epochs=1,
        device=torch.device("cpu"),
        save_dir=None,
        start_epoch=1,
        clip_grad=1,
        noise_std=25,
        verbose=True,
        val_freq=1,
        save_freq=1,
        epoch_fun=None):
    """Train the network to fit the training data."""
    print(f"fit: Using device {device}")

    print("Saving initialization to 0.ckpt")
        # Ensure noise_std is a tuple
    if not isinstance(noise_std, (list, tuple)):
        noise_std = (noise_std, noise_std)
    ckpt_path = os.path.join(save_dir, '0.ckpt')  # Path for the initial checkpoint
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists
    save_ckpt(ckpt_path, net, 0, opt, sched)  # Save initial checkpoint

    epoch = start_epoch
    while epoch < start_epoch + epochs:
        for phase in ['train', 'val', 'test']:
            if phase == 'val' and epoch % val_freq != 0:
                continue
            if phase == 'test' and epoch != epochs:
                continue
            net.train() if phase == 'train' else net.eval()  # Set model to train or eval mode
            dataloader = loaders[phase]  # Get the appropriate dataloader for the phase
            if dataloader is None:
                continue
            phase_nstd = (noise_std[0] + noise_std[1]) / 2.0 if phase in ['val', 'test'] else noise_std

            t = tqdm(iter(dataloader), desc=f"{phase.upper()}-E{epoch}", dynamic_ncols=True)  # Progress bar
            psnr = 0
            total_loss = 0

            for itern, batch in enumerate(t):
                batch = async_prefetch_to_gpu(batch, non_blocking=True)  # Move batch to the specified device asynchronously
                opt.zero_grad()  # Clear gradients
                
                with torch.set_grad_enabled(phase == 'train'):  # Enable gradients if training
                    B, C, D, H, W = batch.shape  # Get batch shape (B, C, Depth, H, W)
                    prev_frame, curr_frame = batch[:, :, 0, :, :], batch[:, :, 1, :, :]
                    prev_frame, curr_frame = prev_frame.to(device), curr_frame.to(device)
                    prev_frame_hat, sigma_n_1 = awgn(prev_frame, phase_nstd)
                    curr_frame_hat, sigma_n_2 = awgn(curr_frame, phase_nstd)

                    # Forward pass for the previous frame
                    z_prev = None  # No previous hidden state for the first frame
                    prev_denoised, z_prev = net(prev_frame_hat, z_prev, sigma_n_1)

                    # Forward pass for the current frame using the previous hidden state
                    curr_denoised, z_curr = net(curr_frame_hat, z_prev, sigma_n_2)

                    # Forward pass again for the previous frame to compute consistency
                    prev_reconstructed, _ = net(prev_frame_hat, z_curr, sigma_n_1)

                    # Calculate loss
                    loss_curr = torch.mean((curr_denoised - curr_frame) ** 2)
                    loss_prev = torch.mean((prev_reconstructed - prev_frame) ** 2)
                    loss = loss_curr + loss_prev

                    if phase == 'train':
                        loss.backward()  # Backpropagation
                        if clip_grad is not None:
                            nn.utils.clip_grad_norm_(net.parameters(), clip_grad)  # Clip gradients
                        opt.step()  # Update weights

                loss_value = loss.item()  # Get loss value
                total_loss += loss_value

                # Compute PSNR
                mse = torch.mean((curr_denoised - curr_frame) ** 2)
                psnr += 10 * torch.log10(1.0 / mse)  # Assuming input range is [0,1]

                if verbose:
                    t.set_postfix_str(f"loss={loss_value:.1e}")

            # Calculate average PSNR and loss
            avg_psnr = psnr / (itern + 1)
            avg_loss = total_loss / (itern + 1)
            print(f"{phase.upper()} Average PSNR: {avg_psnr:.3f} dB, Loss: {avg_loss:.3e}")

            # Save PSNR to file
            with open(os.path.join(save_dir, f'{phase}.txt'), 'a') as psnr_file:
                psnr_file.write(f'{avg_psnr:.3f}, ')

        if sched is not None:
            sched.step()  # Step the learning rate scheduler

        if epoch % save_freq == 0:
            ckpt_path = os.path.join(save_dir, f'net_epoch_{epoch}.ckpt')  # Path for checkpoint
            print(f'Saving checkpoint: {ckpt_path}')
            save_ckpt(ckpt_path, net, epoch, opt, sched)  # Save checkpoint

            if epoch_fun is not None:
                epoch_fun(epoch)  # Call the epoch function

        epoch += 1  # Increment epoch


def init_model(args, device=torch.device("cpu")):
    """Initialize model, optimizer, and scheduler; load from checkpoint if available."""
    model_args, train_args, paths = [args[item] for item in ['model', 'train', 'paths']]
    net = CDLNet_CSR(**model_args)  # Initialize the model
    net.to(device)  # Move model to the specified device

    opt = torch.optim.Adam(net.parameters(), **train_args['opt'])  # Initialize optimizer
    sched = torch.optim.lr_scheduler.StepLR(opt, **train_args['sched'])  # Initialize scheduler
    ckpt_path = paths['ckpt']

    if ckpt_path is not None:
        print(f"Initializing network from {ckpt_path}...")
        net, opt, sched, epoch0 = load_ckpt(ckpt_path, net, opt, sched)  # Load checkpoint
    else:
        epoch0 = 0  # Start from epoch 0 if no checkpoint

    print("Current learning rates:")
    for param_group in opt.param_groups:
        print(param_group['lr'])  # Print current learning rates

    return net, opt, sched, epoch0


def save_ckpt(path, net=None, epoch=None, opt=None, sched=None):
    """Save checkpoint."""
    def getSD(obj):
        return obj.state_dict() if obj is not None else None  # Get state dict if the object exists
    torch.save({
        'epoch': epoch,
        'net_state_dict': getSD(net),
        'opt_state_dict': getSD(opt),
        'sched_state_dict': getSD(sched)
    }, path)
    print(f"Checkpoint saved to {path}")


def load_ckpt(path, net=None, opt=None, sched=None):
    """Load checkpoint."""
    ckpt = torch.load(path, map_location=torch.device('cpu'))  # Load the checkpoint
    def setSD(obj, name):
        if obj is not None and f"{name}_state_dict" in ckpt:
            print(f"Loading {name} state_dict...")
            obj.load_state_dict(ckpt[f"{name}_state_dict"])  # Load state dict
        return obj

    net = setSD(net, 'net')  # Load network state dict
    opt = setSD(opt, 'opt')  # Load optimizer state dict
    sched = setSD(sched, 'sched')  # Load scheduler state dict
    epoch = ckpt.get('epoch', 0)  # Get starting epoch
    print(f"Loaded checkpoint from {path}, starting from epoch {epoch}")
    return net, opt, sched, epoch

def save_args(args, ckpt=True):
    """Save the parameter dictionary to a file and optionally write to checkpoint."""
    save_path = args['paths']['save']
    if ckpt:
        ckpt_path = os.path.join(save_path, f"net.ckpt")
        args['paths']['ckpt'] = ckpt_path  # Update checkpoint path
    with open(os.path.join(save_path, "args.json"), "w") as outfile:
        json.dump(args, outfile, indent=4, sort_keys=True)  # Save arguments as JSON
    print(f"Arguments saved to {os.path.join(save_path, 'args.json')}")

if __name__ == "__main__":
    """Load parameter dictionary from JSON file and pass it to the main function."""
    if len(sys.argv) < 2:
        print('Error: Usage: train.py [path/to/arg_file.json]')
        sys.exit(1)
    args_file = open(sys.argv[1])
    args = json.load(args_file)  # Load arguments from JSON file
    args_file.close()
    main(args)  # Run the main function