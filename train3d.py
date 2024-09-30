import os
import sys
import json
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
from model.net import CDLNetVideo  # Updated to import only CDLNetVideo
from data3d import get_fit_loaders
from utils import awgn3d, gen_bayer_mask3d

def main(args):
    """Load data, initialize the model, and train the model based on the argument dictionary.
    """
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Number of GPUs available: {ngpu}")
    print(f"Using device: {device}")

    model_args, train_args, paths = [args[item] for item in ['model', 'train', 'paths']]
    loaders = get_fit_loaders(**train_args['loaders'])
    net, opt, sched, epoch0 = init_model(args, device=device)

    fit(net, 
        opt, 
        loaders,
        sched       = sched,
        save_dir    = paths['save'],
        start_epoch = epoch0 + 1,
        device      = device,
        **train_args['fit'],
        epoch_fun = lambda epoch_num: save_args(args, epoch_num))

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
        backtrack_thresh=1):
    """Train the network to fit the training data.
    """
    print(f"fit: using device {device}")

    if not isinstance(noise_std, (list, tuple)):
        noise_std = (noise_std, noise_std)

    print("Saving initialization to 0.ckpt")

    ckpt_path = os.path.join(save_dir, '0.ckpt')
    save_ckpt(ckpt_path, net, 0, opt, sched)

    top_psnr = {"train": 0, "val": 0, "test": 0}  # Used for backtracking
    epoch = start_epoch

    while epoch < start_epoch + epochs:
        for phase in ['train', 'val', 'test']:
            net.train() if phase == 'train' else net.eval()
            if epoch != epochs and phase == 'test':
                continue
            if phase == 'val' and epoch % val_freq != 0:
                continue
            if phase in ['val', 'test']:
                phase_nstd = (noise_std[0] + noise_std[1]) / 2.0
            else:
                phase_nstd = noise_std
            psnr = 0

            t = tqdm(iter(loaders[phase]), desc=phase.upper()+'-E'+str(epoch), dynamic_ncols=True)

        for itern, batch in enumerate(t):
            batch = batch.to(device)
            mask = gen_bayer_mask3d(batch) if demosaic else 1
            noisy_batch, sigma_n = awgn3d(batch, phase_nstd)
            obsrv_batch = mask * noisy_batch
            opt.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                batch_hat, _ = net(obsrv_batch, sigma_n, mask=mask)

                if mcsure and phase == "train":
                    h = 1e-3
                    b = torch.randn_like(obsrv_batch)
                    batch_hat_b, _ = net(obsrv_batch.clone() + h * b, sigma_n, mask=mask)
                    div = 2.0 * torch.mean(((sigma_n / 255.0) ** 2) * b * (batch_hat_b - batch_hat)) / h
                    loss = torch.mean((obsrv_batch - batch_hat) ** 2) + div
                else:
                    loss = torch.mean((batch - batch_hat) ** 2)

                if phase == 'train':
                    loss.backward()
                    if clip_grad is not None:
                        nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
                    opt.step()
                    net.project()
            loss = loss.item()


            if verbose:
                total_norm = grad_norm(net.parameters())
                t.set_postfix_str(f"loss={loss:.1e}|gnorm={total_norm:.1e}")
            psnr = psnr - 10 * np.log10(loss)

            psnr = psnr/(itern+1)
            print(f"{phase.upper()} PSNR: {psnr:.3f} dB")


            if psnr > top_psnr[phase]:
                top_psnr[phase] = psnr
            # Backtracking check
            elif (psnr + backtrack_thresh < top_psnr[phase]) or np.isnan(loss) or np.isinf(loss):
                break

            with open(os.path.join(save_dir, f'{phase}.txt'), 'a') as psnr_file:
                psnr_file.write(f'{psnr:.3f}, ')

        # Handle backtracking logic
        if (psnr + backtrack_thresh < top_psnr[phase]) or np.isnan(loss) or np.isinf(loss):
            ckpt_path = os.path.join(save_dir, 'net.ckpt')
            if epoch <= save_freq:
                ckpt_path = os.path.join(save_dir, '0.ckpt')
            print(f"Loss has diverged. Backtracking to {ckpt_path} ...")

            with open(os.path.join(save_dir, f'backtrack.txt'), 'a') as psnr_file:
                psnr_file.write(f'{epoch}  ')

            if epoch % save_freq == 0:
                epoch = epoch - save_freq
            else:
                epoch = epoch - epoch % save_freq

            old_lr = np.array(getlr(opt))
            net, _, _, _ = load_ckpt(ckpt_path, net, opt, sched)
            new_lr = old_lr * 0.8
            setlr(opt, new_lr)
            print("Updated Learning Rate(s):", new_lr)
            epoch = epoch + 1
            continue

        if sched is not None:
            sched.step()
            if hasattr(sched, "step_size") and epoch % sched.step_size == 0:
                print("Updated Learning Rate(s): ")
                print(getlr(opt))

        if epoch % save_freq == 0:
            ckpt_path = os.path.join(save_dir, 'net.ckpt')
            print('Checkpoint: ' + ckpt_path)
            save_ckpt(ckpt_path, net, epoch, opt, sched)

            if epoch_fun is not None:
                epoch_fun(epoch)

        epoch = epoch + 1

def grad_norm(params):
    """Calculate the norm of the gradients for a mini-batch.
    """
    total_norm = 0
    for p in params:
        param_norm = torch.tensor(0.0, device=p.device)
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def getlr(opt):
    """Retrieve the current learning rate of the optimizer.
    """
    return [pg['lr'] for pg in opt.param_groups]

def setlr(opt, lr):
    """Set the learning rate of the optimizer.
    """
    if not isinstance(lr, (list, np.ndarray)):
        lr = [lr for _ in range(len(opt.param_groups))]
    for (i, pg) in enumerate(opt.param_groups):
        pg['lr'] = lr[i]

def init_model(args, device=torch.device("cpu")):
    """Return the model, optimizer, and scheduler, and initialize from a checkpoint if available.
    """
    model_args, train_args, paths = [args[item] for item in ['model', 'train', 'paths']]
    
    # Initialize the CDLNetVideo model directly
    net = CDLNetVideo(**model_args)

    net.to(device)

    opt = torch.optim.Adam(net.parameters(), **train_args['opt'])     
    sched = torch.optim.lr_scheduler.StepLR(opt, **train_args['sched'])
    ckpt_path = paths['ckpt']

    if ckpt_path is not None:
        print(f"Initializing net from {ckpt_path} ...")
        net, opt, sched, epoch0 = load_ckpt(ckpt_path, net, opt, sched)
    else:
        epoch0 = 0

    print("Current Learning Rate(s):")
    for param_group in opt.param_groups:
        print(param_group['lr'])

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total Number of Parameters: {total_params:,}")

    print(f"Using save directory: {paths['save']} ...")
    os.makedirs(paths['save'], exist_ok=True)
    return net, opt, sched, epoch0

def save_ckpt(path, net=None, epoch=None, opt=None, sched=None):
    """Save a checkpoint.
    Saves the state dictionaries of net, optimizer, scheduler, and the epoch number to the specified path.
    """
    def getSD(obj):
        return obj.state_dict() if obj is not None else None
    torch.save({
        'epoch': epoch,
        'net_state_dict': getSD(net),
        'opt_state_dict': getSD(opt),
        'sched_state_dict': getSD(sched)
    }, path)
    print(f"Checkpoint saved to {path}")

def load_ckpt(path, net=None, opt=None, sched=None):
    """Load a checkpoint.
    Loads the state dictionaries of net, optimizer, scheduler, and epoch number from the specified path.
    """
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    def setSD(obj, name):
        if obj is not None and f"{name}_state_dict" in ckpt:
            print(f"Loading {name} state_dict...")
            obj.load_state_dict(ckpt[f"{name}_state_dict"])
        return obj

    net = setSD(net, 'net')
    opt = setSD(opt, 'opt')
    sched = setSD(sched, 'sched')
    epoch = ckpt.get('epoch', 0)
    print(f"Checkpoint loaded from {path}, starting at epoch {epoch}")
    return net, opt, sched, epoch

def save_args(args, ckpt=True):
    """Save the argument dictionary to a file and optionally write to a checkpoint.
    """
    save_path = args['paths']['save']
    if ckpt:
        ckpt_path = os.path.join(save_path, f"net.ckpt")
        args['paths']['ckpt'] = ckpt_path
    with open(os.path.join(save_path, "args.json"), "w") as outfile:
        json.dump(args, outfile, indent=4, sort_keys=True)
    print(f"Arguments saved to {os.path.join(save_path, 'args.json')}")

if __name__ == "__main__":
    """Load the argument dictionary from a JSON file and pass it to main.
    """
    if len(sys.argv) < 2:
        print('ERROR: usage: train.py [path/to/arg_file.json]')
        sys.exit(1)
    args_file = open(sys.argv[1])
    args = json.load(args_file)
    pprint(args)
    args_file.close()
    main(args)