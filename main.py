from data.dataset import ColorHintDataset

import torch
import torch.utils.data as data
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0))), 0, 1) * 255.0
    return image_numpy.astype(imtype)


# Change to your data root directory
root_path = "./datasets"
# Depend on runtime setting
use_cuda = True

train_dataset = ColorHintDataset(root_path, 256, "train")
train_dataloader = data.DataLoader(train_dataset, batch_size=4, shuffle=True)



# import ssim
# import torch.nn.functional as F

# ssim_loss = ssim.SSIM(mul=1000)
# l1_loss = torch.nn.L1Loss()

from utils.MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
from models.model import AttentionR2Unet


model = AttentionR2Unet()
print(model)


for i, data in enumerate(tqdm.tqdm(train_dataloader)):
    if use_cuda:
        l = data["l"].to('cuda')
        ab = data["ab"].to('cuda')
        hint = data["hint"].to('cuda')
    else:
        l = data["l"]
        ab = data["ab"]
        hint = data["hint"]

    gt_image = torch.cat((l, ab), dim=1)
    hint_image = torch.cat((l, hint), dim=1)

    gt_np = tensor2im(gt_image)
    hint_np = tensor2im(hint_image)

    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_LAB2RGB)
    hint_bgr = cv2.cvtColor(hint_np, cv2.COLOR_LAB2RGB)



    # Loss func.

    # ssim_loss_val = ssim_loss(gt_image, hint_image)


    # l1_loss_val = l1_loss(gt_image, hint_image)

    # print("ssim_loss_val", ssim_loss_val)
    # print("l1_loss", l1_loss_val)

    # a = 0.84
    # L_mix = a * L_ms-ssim + (1-a) * L1 * Gaussian_L1

    ms_ssim_l1_loss = MS_SSIM_L1_LOSS(alpha=0.84)

    loss = ms_ssim_l1_loss(gt_image, hint_image)







    # epoch (training / val)

    # test code




    plt.figure(1)
    plt.imshow(gt_bgr)
    print(gt_bgr.shape)
    plt.figure(2)
    plt.imshow(hint_bgr)
    print(hint_bgr.shape)
    plt.show()

    input()

    prev_img = gt_image
