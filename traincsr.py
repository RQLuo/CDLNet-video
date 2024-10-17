import os
import sys
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import necessary functions and classes
from datafastmri import get_fit_loaders
from model.net import (
    CDLNet_CSRf2, CDLNet_CSR, CDLNetVideo, CDLNet, GDLNet, DnCNN, FFDNet
)
from utils import async_prefetch_to_gpu, awgn, awgn3d  # Ensure awgn3d is imported

def main(args):
    """Load data, initialize the model, and train it based on the parameter dictionary."""
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Number of available GPUs: {ngpu}")
    print(f"Using device: {device}")

    # Extract arguments
    model_args = args['model']
    train_args = args['train']
    paths = args['paths']

    # Load data
    loaders = get_fit_loaders(**train_args['loaders'])

    # Initialize model and optimizer
    net, opt, sched, epoch0, model_type = init_model(args, device=device)

    # Train the model
    fit(
        net,
        opt,
        loaders,
        sched=sched,
        save_dir=paths['save'],
        start_epoch=epoch0 + 1,
        model_type=model_type,
        device=device,
        clip_grad=train_args['fit'].get('clip_grad', 1),
        **train_args['fit'],
        epoch_fun=lambda epoch_num: save_args(args, ckpt=True)
    )

def fit(
    net,
    opt,
    loaders,
    sched=None,
    epochs=1,
    device=torch.device("cpu"),
    save_dir=None,
    start_epoch=1,
    model_type=None,
    clip_grad=1,
    noise_std=25,
    verbose=True,
    val_freq=1,
    save_freq=1,
    epoch_fun=None,
    mcsure=False,
    mask=None
):
    """Train the network to fit the training data."""
    print(f"fit: Using device {device}")

    # Ensure noise_std is a tuple
    if not isinstance(noise_std, (list, tuple)):
        noise_std = (noise_std, noise_std)

    # Save initial checkpoint
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, '0.ckpt')
    save_ckpt(ckpt_path, net, 0, opt, sched)

    for epoch in range(start_epoch, start_epoch + epochs):
        for phase in ['train', 'val', 'test']:
            if phase == 'val' and epoch % val_freq != 0:
                continue
            if phase == 'test' and epoch != epochs:
                continue

            net.train() if phase == 'train' else net.eval()
            dataloader = loaders.get(phase)
            if dataloader is None:
                continue

            # Set noise standard deviation for the phase
            phase_nstd = (
                (noise_std[0] + noise_std[1]) / 2.0
                if phase in ['val', 'test'] else noise_std
            )

            t = tqdm(dataloader, desc=f"{phase.upper()}-E{epoch}", dynamic_ncols=True)
            psnr = 0
            total_loss = 0

            for itern, batch in enumerate(t):
                batch = async_prefetch_to_gpu(batch, non_blocking=True)

                # Train model and get loss and MSE
                net, opt, loss, mse = train_model(
                    model_type,
                    batch,
                    phase_nstd,
                    mcsure,
                    net,
                    opt,
                    mask,
                    phase,
                    device,
                    clip_grad
                )

                loss_value = loss.item()
                total_loss += loss_value

                # Compute PSNR
                psnr += 10 * np.log10(1.0 / mse.item())

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
            sched.step()

        if epoch % save_freq == 0:
            ckpt_path = os.path.join(save_dir, f'net_epoch_{epoch}.ckpt')
            print(f'Saving checkpoint: {ckpt_path}')
            save_ckpt(ckpt_path, net, epoch, opt, sched)

            if epoch_fun is not None:
                epoch_fun(epoch)

def train_model(
    model_type,
    batch,
    phase_nstd,
    mcsure,
    net,
    opt,
    mask,
    phase,
    device,
    clip_grad
):
    """Train the model for one iteration and return loss and MSE."""
    if model_type in ["CDLNet", "GDLNet", "DnCNN", "FFDNet"]:
        batch = batch.to(device)
        batch = batch.permute(2, 1, 3, 4, 0).squeeze(-1)
        noisy_batch, sigma_n = awgn(batch, phase_nstd)
        obsrv_batch = noisy_batch
        opt.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            batch_hat, _ = net(obsrv_batch, sigma_n)

            # Supervised or unsupervised (MCSURE) loss during training
            if mcsure and phase == "train":
                h = 1e-3
                b = torch.randn_like(obsrv_batch)
                batch_hat_b, _ = net(obsrv_batch.clone() + h * b, sigma_n)
                div = 2.0 * torch.mean(
                    ((sigma_n / 255.0) ** 2) * b * (batch_hat_b - batch_hat)
                ) / h
                loss = torch.mean((obsrv_batch - batch_hat) ** 2) + div
            else:
                loss = torch.mean((batch - batch_hat) ** 2)

            if phase == 'train':
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
                opt.step()

            mse = torch.mean((batch_hat - batch) ** 2)

    elif model_type == "CDLNet_CSR":
        batch = batch.to(device)
        B, C, D, H, W = batch.shape
        prev_frame, curr_frame = batch[:, :, 0, :, :], batch[:, :, 1, :, :]
        prev_frame_hat, sigma_n_1 = awgn(prev_frame, phase_nstd)
        curr_frame_hat, sigma_n_2 = awgn(curr_frame, phase_nstd)
        opt.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            z_curr = None  # No previous hidden state for the first frame
            for _ in range(2):
                prev_denoised, z_prev = net(prev_frame_hat, z_curr, sigma_n_1)
                curr_denoised, z_curr = net(curr_frame_hat, z_prev, sigma_n_2)

            # MCSURE is not implemented for CDLNet_CSR; use supervised loss
            loss_prev = torch.mean((prev_denoised - prev_frame) ** 2)
            loss_curr = torch.mean((curr_denoised - curr_frame) ** 2)
            loss = loss_prev + loss_curr

            if phase == 'train':
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
                opt.step()

            mse = (loss_prev + loss_curr) / 2.0

    elif model_type == "CDLNetVideo":
        batch = batch.to(device)
        noisy_batch, sigma_n = awgn3d(batch, phase_nstd)
        obsrv_batch = noisy_batch
        opt.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            batch_hat, _ = net(obsrv_batch, sigma_n)

            # Supervised or unsupervised (MCSURE) loss during training
            if mcsure and phase == "train":
                h = 1e-3
                b = torch.randn_like(obsrv_batch)
                batch_hat_b, _ = net(obsrv_batch.clone() + h * b, sigma_n)
                div = 2.0 * torch.mean(
                    ((sigma_n / 255.0) ** 2) * b * (batch_hat_b - batch_hat)
                ) / h
                loss = torch.mean((obsrv_batch - batch_hat) ** 2) + div
            else:
                loss = torch.mean((batch - batch_hat) ** 2)

            if phase == 'train':
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
                opt.step()

            mse = torch.mean((batch_hat - batch) ** 2)
    elif model_type == "CDLNet_CSRf2":
        batch = batch.to(device)
        B, C, D, H, W = batch.shape
        prev_frame, curr_frame, after_frame  = batch[:, :, 0, :, :], batch[:, :, 1, :, :], batch[:, :, 2, :, :]
        prev_frame_hat, sigma_n_1 = awgn(prev_frame, phase_nstd)
        curr_frame_hat, sigma_n_2 = awgn(curr_frame, phase_nstd)
        after_frame_hat, sigma_n_3 = awgn(after_frame, phase_nstd)
        opt.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            prev_denoised, z_prev = net(prev_frame_hat, None, None, sigma_n_1)
            curr_denoised, z_curr = net(curr_frame_hat, z_prev, None, sigma_n_2)
            after_denoised, z_after = net(after_denoised, z_prev, None, sigma_n_3)
            curr_denoised, z_curr = net(curr_frame_hat, z_prev, z_after, sigma_n_2)
            prev_denoised, z_prev = net(prev_frame_hat, None, z_after, sigma_n_1)
            loss_prev = torch.mean((prev_denoised - prev_frame) ** 2)
            loss_curr = torch.mean((curr_denoised - curr_frame) ** 2)
            loss_after = torch.mean((after_denoised - after_frame) ** 2)
            loss = loss_prev + loss_curr + loss_after

            if phase == 'train':
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
                opt.step()

            mse = (loss_prev + loss_curr + loss_after) / 3.0
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not implemented.")

    return net, opt, loss, mse

def init_model(args, device=torch.device("cpu")):
    """Initialize model, optimizer, and scheduler; load from checkpoint if available."""
    model_type = args['type']
    model_args = args['model']
    train_args = args['train']
    paths = args['paths']

    # Initialize the model based on its type
    if model_type == "CDLNet":
        net = CDLNet(**model_args)
    elif model_type == "GDLNet":
        net = GDLNet(**model_args)
    elif model_type == "DnCNN":
        net = DnCNN(**model_args)
    elif model_type == "FFDNet":
        net = FFDNet(**model_args)
    elif model_type == "CDLNet_CSR":
        net = CDLNet_CSR(**model_args)
    elif model_type == "CDLNet_CSRf2":
        net = CDLNet_CSRf2(**model_args)
    elif model_type == "CDLNetVideo":
        net = CDLNetVideo(**model_args)
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not implemented.")

    net.to(device)
    opt = torch.optim.Adam(net.parameters(), **train_args['opt'])
    sched = torch.optim.lr_scheduler.StepLR(opt, **train_args['sched'])
    ckpt_path = paths.get('ckpt')

    if ckpt_path is not None:
        print(f"Initializing network from {ckpt_path}...")
        net, opt, sched, epoch0 = load_ckpt(ckpt_path, net, opt, sched)
    else:
        epoch0 = 0

    print("Current learning rates:")
    for param_group in opt.param_groups:
        print(param_group['lr'])

    return net, opt, sched, epoch0, model_type

def save_ckpt(path, net=None, epoch=None, opt=None, sched=None):
    """Save checkpoint."""
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict() if net else None,
        'opt_state_dict': opt.state_dict() if opt else None,
        'sched_state_dict': sched.state_dict() if sched else None
    }, path)
    print(f"Checkpoint saved to {path}")

def load_ckpt(path, net=None, opt=None, sched=None):
    """Load checkpoint."""
    ckpt = torch.load(path, map_location=torch.device('cpu'))

    def set_state(obj, state_dict_key):
        if obj and state_dict_key in ckpt:
            obj.load_state_dict(ckpt[state_dict_key])
        return obj

    net = set_state(net, 'net_state_dict')
    opt = set_state(opt, 'opt_state_dict')
    sched = set_state(sched, 'sched_state_dict')
    epoch = ckpt.get('epoch', 0)
    print(f"Loaded checkpoint from {path}, starting from epoch {epoch}")
    return net, opt, sched, epoch

def save_args(args, ckpt=True):
    """Save the parameter dictionary to a file and optionally write to checkpoint."""
    save_path = args['paths']['save']
    if ckpt:
        ckpt_path = os.path.join(save_path, "net.ckpt")
        args['paths']['ckpt'] = ckpt_path
    with open(os.path.join(save_path, "args.json"), "w") as outfile:
        json.dump(args, outfile, indent=4, sort_keys=True)
    print(f"Arguments saved to {os.path.join(save_path, 'args.json')}")

if __name__ == "__main__":
    """Load parameter dictionary from JSON file and pass it to the main function."""
    if len(sys.argv) < 2:
        print('Error: Usage: train.py [path/to/arg_file.json]')
        sys.exit(1)
    with open(sys.argv[1]) as args_file:
        args = json.load(args_file)
    main(args)
