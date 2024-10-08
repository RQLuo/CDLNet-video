import os
import sys
import json
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
from model.net import CDLNetVideo  # Ensure the model is imported correctly
from datafastmri import get_fit_loaders
from utils import awgn3d, gen_bayer_mask
from loss import CombinedLossWithSSIM

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
        demosaic=False,
        verbose=True,
        val_freq=1,
        save_freq=1,
        epoch_fun=None,
        mcsure=False,
        combmse=False,
        backtrack_thresh=1):
    """Train the network to fit the training data."""
    print(f"fit: Using device {device}")

    # Ensure noise_std is a tuple
    if not isinstance(noise_std, (list, tuple)):
        noise_std = (noise_std, noise_std)

    print("Saving initialization to 0.ckpt")

    ckpt_path = os.path.join(save_dir, '0.ckpt')  # Path for the initial checkpoint
    save_ckpt(ckpt_path, net, 0, opt, sched)  # Save initial checkpoint

    top_psnr = {"train": 0, "val": 0, "test": 0}  # For tracking the best PSNR values
    epoch = start_epoch
    if combmse:
        loss_fn = CombinedLossWithSSIM(alpha=1.0, beta=0.01, gamma=0.1).to(device)
    # Training loop
    while epoch < start_epoch + epochs:
        for phase in ['train', 'val', 'test']:
            # Skip the test phase if it's not the final epoch
            if phase == 'test' and epoch != epochs:
                continue
            # Skip the validation phase if not the right epoch
            if phase == 'val' and epoch % val_freq != 0:
                continue

            net.train() if phase == 'train' else net.eval()  # Set model to train or eval mode

            # Determine noise standard deviation for the current phase
            phase_nstd = (noise_std[0] + noise_std[1]) / 2.0 if phase in ['val', 'test'] else noise_std
            psnr = 0

            dataloader = loaders[phase]  # Get the appropriate dataloader for the phase
            if dataloader is None:
                continue

            t = tqdm(iter(dataloader), desc=f"{phase.upper()}-E{epoch}", dynamic_ncols=True)  # Progress bar

            # Iterate over batches
            for itern, batch in enumerate(t):
                batch = batch.to(device)  # Move batch to the specified device
                mask = gen_bayer_mask(batch) if demosaic else 1
                noisy_batch, sigma_n = awgn3d(batch, phase_nstd)
                obsrv_batch = mask * noisy_batch
                opt.zero_grad()  # Clear gradients

                with torch.set_grad_enabled(phase == 'train'):  # Enable gradients if training
                    batch_hat, _ = net(noisy_batch, sigma_n)  # Forward pass
                    # supervised or unsupervised (MCSURE) loss during training
                    if mcsure and phase == "train":
                        h = 1e-3
                        b = torch.randn_like(obsrv_batch)
                        batch_hat_b, _ = net(obsrv_batch.clone() + h*b, sigma_n, mask=mask)
                        # assume you have a good estimator for sigma_n
                        div = 2.0*torch.mean(((sigma_n/255.0)**2)*b*(batch_hat_b-batch_hat)) / h
                        loss = torch.mean((obsrv_batch - batch_hat)**2) + div
                    elif combmse and phase == "train":
                        loss = loss_fn(batch_hat, batch)
                    else:    
                        loss = torch.mean((batch - batch_hat)**2)

                    if phase == 'train':
                        loss.backward()  # Backpropagation
                        if clip_grad is not None:
                            nn.utils.clip_grad_norm_(net.parameters(), clip_grad)  # Clip gradients
                        opt.step()  # Update weights

                loss_value = loss.item()  # Get loss value

                if verbose:
                    total_norm = grad_norm(net.parameters())  # Calculate gradient norm
                    t.set_postfix_str(f"loss={loss_value:.1e}|gnorm={total_norm:.1e}")

                # Compute PSNR
                psnr += 10 * np.log10(1.0 / loss_value)  # Assuming input range is [0,1]

            # Calculate average PSNR
            avg_psnr = psnr / (itern + 1)
            print(f"{phase.upper()} Average PSNR: {avg_psnr:.3f} dB")

            if avg_psnr > top_psnr[phase]:
                top_psnr[phase] = avg_psnr  # Update top PSNR if current is better

            # Save PSNR to file
            with open(os.path.join(save_dir, f'{phase}.txt'), 'a') as psnr_file:
                psnr_file.write(f'{avg_psnr:.3f}, ')

        if sched is not None:
            sched.step()  # Step the learning rate scheduler
            if hasattr(sched, "step_size") and epoch % sched.step_size == 0:
                print("Updating learning rate:")
                print(getlr(opt))

        if epoch % save_freq == 0:
            ckpt_path = os.path.join(save_dir, f'net_epoch_{epoch}.ckpt')  # Path for checkpoint
            print(f'Saving checkpoint: {ckpt_path}')
            save_ckpt(ckpt_path, net, epoch, opt, sched)  # Save checkpoint

            if epoch_fun is not None:
                epoch_fun(epoch)  # Call the epoch function

        epoch += 1  # Increment epoch

def grad_norm(params):
    """Compute the norm of gradients."""
    total_norm = 0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()  # L2 norm
            total_norm += param_norm ** 2
    return total_norm ** 0.5  # Return the overall norm

def getlr(opt):
    """Get the current learning rate of the optimizer."""
    return [pg['lr'] for pg in opt.param_groups]  # Return learning rates of all parameter groups

def setlr(opt, lr):
    """Set the learning rate of the optimizer."""
    if not isinstance(lr, (list, np.ndarray)):
        lr = [lr for _ in range(len(opt.param_groups))]  # Create a list of learning rates
    for (i, pg) in enumerate(opt.param_groups):
        pg['lr'] = lr[i]  # Set the learning rate for each group

def init_model(args, device=torch.device("cpu")):
    """Initialize model, optimizer, and scheduler; load from checkpoint if available."""
    model_args, train_args, paths = [args[item] for item in ['model', 'train', 'paths']]
    
    # Ensure all required model parameters are included in JSON
    required_args = ['adaptive', 'K', 'M', 'C', 'P', 's', 't0', 'init']
    for arg in required_args:
        if arg not in model_args:
            raise ValueError(f"Missing '{arg}' in model parameters")

    net = CDLNetVideo(**model_args)  # Initialize the model
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

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)  # Count trainable parameters
    print(f"Total trainable parameters: {total_params:,}")

    print(f"Using save directory: {paths['save']} ...")
    os.makedirs(paths['save'], exist_ok=True)  # Create save directory if it doesn't exist
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
    pprint(args)
    args_file.close()
    main(args)  # Run the main function
