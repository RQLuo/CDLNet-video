#!/usr/bin/env python3
import os
import sys
import json
import copy
import time
from pprint import pprint
from tqdm import tqdm
import numpy as np
from numpy.fft import fftshift, fft2
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import model
import model.nle
import utils, data3d, train3d  # Ensure these modules are compatible with 3D data

import argparse

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("args_fn", type=str, help="Path to args.json file.", default="args.json")
parser.add_argument("--test", type=str, help="Run model over specified test set (provided path to video dir).", default=None)
parser.add_argument("--dictionary", action="store_true", help="Save image of final synthesis dictionary and magnitude freq-response.")
parser.add_argument("--passthrough", type=str, help="Example passthrough of model.", default=None)
parser.add_argument("--noise_level", type=int, nargs='*', help="Input noise-level(s) on [0,255] range. Single value required for --passthrough. If --test is used, multiple values can be specified.", default=[-1])
parser.add_argument("--blind", type=str, default=None, choices=["MAD", "PCA"], help="Blind noise-level estimation algorithm.")
parser.add_argument("--save", action="store_true", help="Save test, intermediate passthrough results to files.")
parser.add_argument("--thresholds", action="store_true", help="Plot network thresholds.")
parser.add_argument("--filters", action="store_true", help="Save network A,B filterbanks.")
parser.add_argument("--save_dir", type=str, help="Where to save analyze results.", default=None)
parser.add_argument("--color", action="store_true", help="Use color images.")
parser.add_argument("--demosaic", action="store_true", help="Demosaicing problem.")

ARGS = parser.parse_args()

def main(model_args):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")

    # Initialize the 3D model
    net, _, _, epoch0 = train3d.init_model(model_args, device=device)
    net.eval()

    # Set save directory
    if ARGS.save_dir is None:
        ARGS.save_dir = model_args['paths']['save']

    # Set noise level
    if len(ARGS.noise_level) == 1:
        ARGS.noise_level = ARGS.noise_level[0]
    if ARGS.noise_level == -1:
        ARGS.noise_level = model_args['train']['fit']['noise_std']

    with torch.no_grad():
        if ARGS.test is not None:
            # Load test data using the 3D DataLoader
            loader = data3d.get_data_loader([ARGS.test], load_color=ARGS.color, test=True, depth=16)
            test(net, loader, noise_level=ARGS.noise_level, blind=ARGS.blind, device=device)

        if ARGS.dictionary:
            dictionary(net)

        if ARGS.passthrough is not None:
            passthrough(net, ARGS.passthrough, ARGS.noise_level, blind=ARGS.blind, demosaic=ARGS.demosaic, device=device, color=ARGS.color)

        if ARGS.thresholds:
            thresholds(net, noise_level=ARGS.noise_level)

        if ARGS.filters:
            filters(net, scale_each=True)

def test(net, loader, noise_level=25, blind=False, device=torch.device('cpu')):
    """ Evaluate net on test-set with 3D video data.
    Processes each video by splitting it into 16-frame batches.
    """
    print("--------- test ---------")
    dset_name = os.path.basename(os.path.dirname(loader.dataset.video_dirs[0]))
    fn = os.path.join(ARGS.save_dir, f"test_{dset_name}_{blind}.txt")

    if not isinstance(noise_level, (range, list, tuple)):
        noise_level = [noise_level]

    if ARGS.save:
        test_noise_dir = os.path.join(ARGS.save_dir, "test_noise")
        test_output_dir = os.path.join(ARGS.save_dir, "test_output")
        os.makedirs(test_noise_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

    for sigma in noise_level:
        print(f"Processing noise level: {sigma}")
        t = tqdm(iter(loader), desc=f"TEST-{sigma}", dynamic_ncols=True)
        psnr_total = 0
        frame_count = 0

        for batch_idx, video in enumerate(t):
            video = video.to(device)  # Shape: (B, C, D, H, W)
            B, C, D, H, W = video.shape

            # Ensure each batch has exactly 16 frames
            if D != 16:
                raise ValueError(f"Expected depth=16, but got depth={D} in batch {batch_idx}")

            # Generate Bayer mask if demosaicing, else use mask=1
            mask = utils.gen_bayer_mask3d(video) if ARGS.demosaic else 1

            # Add AWGN noise
            noisy_video, s = utils.awgn3d(video, sigma)

            # Apply mask
            noisy_video = mask * noisy_video

            # Handle adaptive noise level estimation
            if net.adaptive:
                if blind:
                    s = 255 * model.nle.noise_level(noisy_video, method=blind)
                    print(f"sigma_hat = {s.flatten().item():.3f}")
                else:
                    print("Using ground truth sigma.")
            else:
                s = None

            # Denoise the video
            denoised_video, _ = net(noisy_video, s, mask=mask)

            # Calculate PSNR for each frame and accumulate
            mse = torch.mean((video - denoised_video) ** 2, dim=[1, 2, 3, 4])  # MSE per video in batch
            psnr = -10 * torch.log10(mse).mean().item()
            psnr_total += psnr
            frame_count += B * D

            # Save noisy and denoised frames if required
            if ARGS.save:
                # Clamp the values to [0,1] for saving
                noisy_clamped = torch.clamp(noisy_video, 0, 1)
                denoised_clamped = torch.clamp(denoised_video, 0, 1)

                for b in range(B):
                    for d_idx in range(D):
                        frame_number = batch_idx * D + d_idx + 1  # Assuming sequential numbering
                        noisy_filename = os.path.join(test_noise_dir, f"noise_{frame_number:05d}.png")
                        denoised_filename = os.path.join(test_output_dir, f"output_{frame_number:05d}.png")

                        # Save each frame individually
                        save_image(noisy_clamped[b, :, d_idx, :, :], noisy_filename)
                        save_image(denoised_clamped[b, :, d_idx, :, :], denoised_filename)

        # Average PSNR over all batches
        avg_psnr = psnr_total / (batch_idx + 1)
        print(f"Average PSNR for sigma={sigma}: {avg_psnr:.3f} dB")

        # Log the results
        with open(fn, 'a') as log_file:
            log_file.write(f"{sigma}, {avg_psnr:.3f}\n")

    print(f"Saved PSNR results to file {fn}")
    print("Testing completed.")

def thresholds(net, noise_level=25):
    """ Plot and save network thresholds. """
    print("--------- thresholds ---------")
    c = 1 if net.adaptive else 0
    tau = torch.cat([
        net.t[k][0:1] + c * (noise_level / 255) * net.t[k][1:2] 
        for k in range(net.K)
    ]).detach()  # Shape: (K, M, 1, 1)

    fig, ax = plt.subplots()
    im = ax.imshow(tau[:, :, 0, 0], cmap='hot', interpolation=None, vmin=0, vmax=tau.max())
    plt.xlabel("Subband (j)")
    plt.ylabel("Iteration (k)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    S = 100
    cbar.set_ticks([0, np.round(S * tau.max() * 0.5) / S, np.floor(S * tau.max() * 1) / S])

    fn = os.path.join(ARGS.save_dir, "tau.png")
    print(f"Saving threshold plot to {fn}...")
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()

    print("Threshold plot saved.")
    print("done.")

def filters(net, scale_each=False):
    """ Saves all network filters. """
    print("--------- filters ---------")
    save_dir = os.path.join(ARGS.save_dir, "filters")
    os.makedirs(save_dir, exist_ok=True)

    # Define how to get filters based on model type
    if isinstance(net, model.net.GDLNet):
        get_filter = lambda C: C.get_filter()
    elif isinstance(net, model.net.CDLNetVideo):
        get_filter = lambda C: C.weight.data
    else:
        raise NotImplementedError("Filter extraction not implemented for this model type.")

    # Extract dictionary filters
    D = get_filter(net.D)
    n = int(np.ceil(np.sqrt(D.shape[0])))

    # Store analysis and synthesis filters
    AL, BL = [], []

    # Determine maximum filter value for consistent scaling
    mmax = 0
    for k in range(net.K):
        AL.append(get_filter(net.A[k]))
        B = get_filter(net.B[k])
        if k == 0:
            B = 0 * B  # Zero out B for the first filterbank
        BL.append(B)

        amax = AL[k].abs().max()
        bmax = BL[k].abs().max()
        if amax > mmax:
            mmax = amax
        if bmax > mmax:
            mmax = bmax

    # Save filterbanks
    for k in range(net.K):
        vr = None if scale_each else (-mmax, mmax)
        Ag = make_grid(AL[k], nrow=n, padding=2, scale_each=scale_each, normalize=True, value_range=vr)

        if k == 0:
            vr = (-1, 1)
        else:
            vr = None if scale_each else (-mmax, mmax)
        Bg = make_grid(BL[k], nrow=n, padding=2, scale_each=scale_each, normalize=True, value_range=vr)

        fn = os.path.join(save_dir, f"AB{k:02d}_{scale_each}.png")
        print(f"Saving filterbank AB{k:02d} to {fn} ...")
        save_image([Ag, Bg], fn, nrow=2, padding=5)

    # Save dictionary filters
    fn = os.path.join(save_dir, f"D_filters_{scale_each}.png")
    print(f"Saving dictionary filters to {fn} ...")
    save_image(D, fn, nrow=n, scale_each=scale_each, normalize=True)
    print("Filterbanks saved.")
    print("done.")

def dictionary(net):
    """ Saves the model's dictionary filters and their frequency responses. """
    print("--------- dictionary ---------")
    if isinstance(net, model.net.CDLNetVideo):
        D = net.D.weight.cpu()
    elif isinstance(net, model.net.GDLNet):
        D = net.D.get_filter().cpu()
    else:
        raise NotImplementedError("Dictionary extraction not implemented for this model type.")

    n = int(np.ceil(np.sqrt(net.M)))

    # Save learned dictionary filters
    fn = os.path.join(ARGS.save_dir, "D_learned.png")
    print(f"Saving learned dictionary to {fn} ...")
    print(f"Dictionary shape: {D.shape}")
    save_image(D, fn, nrow=n, padding=2, scale_each=True, normalize=True)

    # Compute and save frequency response
    X = torch.tensor(fftshift(fft2(D.numpy(), s=(64, 64)), axes=(-2, -1)))
    freq_response = X.abs()

    fn = os.path.join(ARGS.save_dir, "freq_response.png")
    print(f"Saving dictionary frequency response to {fn} ...")
    save_image(freq_response, fn, nrow=n, normalize=True, scale_each=True, padding=10, pad_value=1)
    print("Dictionary and frequency response saved.")
    print("done.")

def passthrough(net, video_path, noise_std, device=torch.device('cpu'), blind=False, color=False, demosaic=False):
    """ Save passthrough of a single video. """
    print("--------- passthrough ---------")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(ARGS.save_dir, f"passthrough_{video_name}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing video: {video_path}...")
    # Load video frames as a 3D tensor
    video = utils.load_video(video_path, gray=not color).to(device)  # Shape: (C, D, H, W)
    B = 1  # Single video

    # Add AWGN noise
    noisy_video, sigma = utils.awgn(video.unsqueeze(0), noise_std)  # Shape: (1, C, D, H, W)
    mask = utils.gen_bayer_mask(noisy_video) if demosaic else 1
    noisy_video = mask * noisy_video

    print(f"Noise std: {sigma}")

    # Handle adaptive noise estimation
    if net.adaptive:
        if blind:
            sigma_est = 255 * model.nle.noise_level(noisy_video, method=blind)
            print(f"Estimated sigma: {sigma_est.flatten().item():.3f}")
        else:
            print("Using ground truth sigma.")
    else:
        sigma_est = None

    # Generate denoised video
    denoised_video, params = net(noisy_video, sigma_est, mask=mask)

    # Calculate PSNR
    mse = torch.mean((video.unsqueeze(0) - denoised_video) ** 2)
    psnr = -10 * torch.log10(mse).item()
    print(f"PSNR: {psnr:.2f} dB")

    if ARGS.save:
        # Clamp values for saving
        noisy_clamped = torch.clamp(noisy_video, 0, 1)
        denoised_clamped = torch.clamp(denoised_video, 0, 1)

        # Save each frame in the video
        for d_idx in range(denoised_clamped.shape[2]):
            frame_number = d_idx + 1
            noisy_filename = os.path.join(save_dir, f"noise_{frame_number:05d}.png")
            denoised_filename = os.path.join(save_dir, f"output_{frame_number:05d}.png")

            save_image(noisy_clamped[0, :, d_idx, :, :], noisy_filename)
            save_image(denoised_clamped[0, :, d_idx, :, :], denoised_filename)

        # Save comparison image (concatenated)
        comparison = torch.cat([noisy_clamped, denoised_clamped, video.unsqueeze(0)], dim=1)  # Shape: (1, 3C, D, H, W)
        comparison = comparison.view(-1, comparison.shape[1], comparison.shape[3], comparison.shape[4])  # Merge depth
        fn = os.path.join(save_dir, "compare.png")
        print(f"Saving comparison image to {fn} ...")
        save_image(comparison, fn, nrow=3, scale_each=False, normalize=False)
    
    # Log PSNR
    fn_log = os.path.join(save_dir, "psnr.txt")
    with open(fn_log, 'w') as log_file:
        log_file.write(f"PSNR: {psnr:.2f} dB\n")

    print("Passthrough completed.")
    print("done.")

if __name__ == "__main__":
    """ Load arguments from json file and command line, then execute main. """
    # Load the provided args.json file
    with open(ARGS.args_fn) as model_args_file:
        model_args = json.load(model_args_file)
    pprint(model_args)
    
    main(model_args)
