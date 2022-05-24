
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch


class SSIM():
    def __init__(self, mul) -> None:
        self.mul = mul
        pass

    def __call__(self, gt, test) -> torch.Tensor:

        # reuse the gaussian kernel with SSIM & MS_SSIM. 
        # ssim_module = SSIM(data_range=255, size_average=True, channel=3) # channel=1 for grayscale images
        ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

        # ssim_loss = 1 - ssim_module(gt_image, gt_image)
        ms_ssim_loss = 1 - ms_ssim_module(gt, test)

        return ms_ssim_loss * self.mul
