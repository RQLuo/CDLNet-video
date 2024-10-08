import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_msssim import ssim

class CombinedLossWithSSIM(nn.Module):
    def __init__(self, alpha=1.0, beta=0.01, gamma=0.1):
        super(CombinedLossWithSSIM, self).__init__()
        self.mse = nn.MSELoss()
        self.vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.alpha = alpha  # MSE loss weight
        self.beta = beta    # Perceptual loss weight
        self.gamma = gamma  # SSIM loss weight

    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        
        perceptual_loss = 0
        num_frames = output.shape[2]
        for t in range(num_frames):
            output_frame = output[:, :, t, :, :]  # [batch_size, 1, H, W]
            target_frame = target[:, :, t, :, :]  # [batch_size, 1, H, W]
            
            # Duplicate channels to convert from 1 to 3 channels
            output_frame_3ch = output_frame.repeat(1, 3, 1, 1)  # [batch_size, 3, H, W]
            target_frame_3ch = target_frame.repeat(1, 3, 1, 1)  # [batch_size, 3, H, W]
            
            # Ensure the duplicated frames are on the same device as VGG
            output_frame_3ch = output_frame_3ch.to(next(self.vgg.parameters()).device)
            target_frame_3ch = target_frame_3ch.to(next(self.vgg.parameters()).device)
            
            output_features = self.vgg(output_frame_3ch)
            target_features = self.vgg(target_frame_3ch)
            perceptual_loss += self.mse(output_features, target_features)
        perceptual_loss /= num_frames

        ssim_loss = 0
        for t in range(num_frames):
            output_frame = output[:, :, t, :, :]  # [batch_size, 1, H, W]
            target_frame = target[:, :, t, :, :]  # [batch_size, 1, H, W]
            
            # If SSIM expects 3 channels, duplicate them as well
            output_frame_3ch = output_frame.repeat(1, 3, 1, 1)
            target_frame_3ch = target_frame.repeat(1, 3, 1, 1)
            
            # Ensure the duplicated frames are on the same device as SSIM
            output_frame_3ch = output_frame_3ch.to(output.device)
            target_frame_3ch = target_frame_3ch.to(target.device)
            
            ssim_val = ssim(output_frame_3ch, target_frame_3ch, data_range=output_frame_3ch.max() - output_frame_3ch.min())
            ssim_loss += (1 - ssim_val)
        ssim_loss /= num_frames

        total_loss = self.alpha * mse_loss + self.beta * perceptual_loss + self.gamma * ssim_loss
        return total_loss
