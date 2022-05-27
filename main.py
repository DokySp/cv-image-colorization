from data.dataset import ColorHintDataset

import torch
import torch.utils.data as data
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

from utils.logger import send_log

from utils.MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
from models.model import AttentionR2Unet

import torch
import torch.nn as nn  # Neural Network -> 객체 추상화

import os
import matplotlib.pyplot as plt

from datetime import datetime

import torch.optim as optim
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from utils.tensor2im import tensor2im

import wandb


log_name = "cv220527-re3-64"
wandb.init(project="cv1", entity="dokysp")

# Change to your data root directory
root_path = "./datasets"
# Depend on runtime setting
use_cuda = True

train_dataset = ColorHintDataset(root_path, 256, "train")
train_dataloader = data.DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = ColorHintDataset(root_path, 256, "val")
val_dataloader = data.DataLoader(val_dataset, batch_size=4, shuffle=False)

# test_dataset = ColorHintDataset(root_path, 256, "test")
# test_dataloader = data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# import ssim
# import torch.nn.functional as F

# ssim_loss = ssim.SSIM(mul=1000)
# l1_loss = torch.nn.L1Loss()


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


def train_1_epoch(model, dataloader, optimizer, criterion):
    # 시각화를 위한 변수
    # confusion_matrix = [[0, 0], [0, 0]]
    total_loss = 0
    iteration = 0

    model.train()  # PyTorch: train, test mode 운영

    for data in tqdm.auto.tqdm(dataloader):

        if use_cuda:
            l = data["l"].to("cuda")
            ab = data["ab"].to("cuda")
            hint = data["hint"].to("cuda")
        else:
            l = data["l"]
            ab = data["ab"]
            hint = data["hint"]

        gt_image = torch.cat((l, ab), dim=1)
        hint_image = torch.cat((l, hint), dim=1)

        optimizer.zero_grad()  # 현재 배치에서 이전 gradient 필요 없음! -> 초기화
        output = model(hint_image).squeeze()  # [Batch, 1] (2치원) -> [Batch] (1차원)

        # y hat, y
        loss = criterion(output, gt_image)

        # back propagation
        loss.backward()

        # Gradient의 learnable parameter update (lr, Adam 안에 기타 변수들 등등)
        optimizer.step()

        

        # ----------

        total_loss += loss.detach()  # detach -> parameter 연산에 사용 X
        wandb.log({"train_each_loss": float(loss.detach().cpu().flatten()[0])})
        iteration += 1

        l.to("cpu")
        ab.to("cpu")
        hint.to("cpu")
        l = ""
        ab = ""
        hint = ""

    #     for i in range(len(label)):
    #         real_class = int(label[i])
    #         pred_class = int(output[i] > 0.5)
    #         confusion_matrix[real_class][pred_class] += 1

    # positive = confusion_matrix[0][0] + confusion_matrix[1][1]
    # negative = confusion_matrix[0][1] + confusion_matrix[1][0]

    # accuracy = positive / (positive + negative)
    total_loss /= iteration
    return total_loss


#
#
#
#
#


def validation_1_epoch(model, dataloader, criterion):
    # 시각화를 위한 변수
    # confusion_matrix = [[0, 0], [0, 0]]
    total_loss = 0
    iteration = 0

    model.eval()  # PyTorch: train, test mode 운영

    for data in tqdm.auto.tqdm(dataloader):

        if use_cuda:
            l = data["l"].to("cuda")
            ab = data["ab"].to("cuda")
            hint = data["hint"].to("cuda")
        else:
            l = data["l"]
            ab = data["ab"]
            hint = data["hint"]

        gt_image = torch.cat((l, ab), dim=1)
        hint_image = torch.cat((l, hint), dim=1)

        output = model(hint_image).squeeze()

        # y hat, y
        loss = criterion(output, gt_image)

        total_loss += loss.detach()  # detach -> parameter 연산에 사용 X
        wandb.log({"val_each_loss": float(loss.detach().cpu().flatten()[0])})
        iteration += 1

        # for i in range(len(label)):
        #     real_class = int(label[i])
        #     pred_class = int(output[i] > 0.5)
        #         confusion_matrix[real_class][pred_class] += 1

        l.to("cpu")
        ab.to("cpu")
        hint.to("cpu")
        l = ""
        ab = ""
        hint = ""

    # positive = confusion_matrix[0][0] + confusion_matrix[1][1]
    # negative = confusion_matrix[0][1] + confusion_matrix[1][0]

    # accuracy = positive / (positive + negative)
    total_loss /= iteration
    return total_loss


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


def main(lrs, epochs, optims, alpha):

    now_time = str(datetime.now())

    # Load Model
    model = AttentionR2Unet(recurrent_iter=2).cuda()

    # Loss func.
    criterion = MS_SSIM_L1_LOSS(alpha=alpha, data_range=255)

    # 옵티마이저
    optimizer = optims(model.parameters(), lr=lrs)  # 학습할 것들을 옵팀에 넘김
    schedular = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.1)


    # 기타 변수들
    train_info = []
    val_info = []
    object_epoch = epochs

    save_path = "./saved_models"
    os.makedirs(save_path, exist_ok=True)
    # output_path = os.path.join(save_path, "basic_model.tar") # 관습적으로 tar을 확장자로 사용
    # output_path = os.path.join(save_path, "validation_model.tar")
    output_path = os.path.join(save_path, "AttentionR2Unet_" + now_time)
    output_path = output_path.replace(" ", "__")
    output_path = output_path.replace("-", "_")

    file_iter = 0

    #
    #
    #
    #
    #

    min_lose = 100000000000000

    for epoch in range(object_epoch):

        wandb.log({"epoch": epoch})




        train_loss = train_1_epoch(
            model, train_dataloader, optimizer=optimizer, criterion=criterion
        )
        train_msg = "[Training] Epoch {}: loss: {}".format(epoch, train_loss)

        print(train_msg)

        train_loss = float(train_loss.detach().cpu().flatten()[0])
        train_info.append(train_loss)

        wandb.log({"train_loss": train_loss})
        send_log(log_name, train_loss)





        # Validation
        with torch.no_grad():  # gradient 계산 X
            val_loss = validation_1_epoch(model, val_dataloader, criterion=criterion)

        val_msg = "[Validation] Epoch: {}, loss: {}".format(epoch, val_loss)

        print(val_msg)

        val_loss = float(val_loss.detach().cpu().flatten()[0])
        val_info.append(val_loss)

        wandb.log({"val_loss": val_loss})
        send_log(log_name, val_loss)
        





        # Update Learning Rate
        new_lr = schedular.get_last_lr()[0]
        wandb.log({"learning_rate": new_lr})
        send_log(log_name, new_lr)
        print("curr_lr", new_lr)
        
        schedular.step()
        new_lr = schedular.get_last_lr()[0]
        print("new_lr", new_lr)







        # 제일 정확한 모델만 저장!
        if min_lose > val_loss:
            min_lose = val_loss
            min_loss_msg = "min_loss: " + str(min_lose)
            loggggg = min_lose

            print(min_loss_msg)
            send_log(log_name, min_loss_msg)
            wandb.log({"min_loss": loggggg})


            torch.save(
                {
                    "memo": "Test",
                    "lrs": lrs,
                    "epochs": epochs,
                    "optims": optims,
                    "alpha": alpha,
                    "loss": min_lose,
                    "state_dict": model.state_dict(),  # 모든 weight 변수이름 / parameter 값들을 가진 dict.
                },
                output_path + "_e" + str(epoch) + ".pth",
            )

            out = pd.DataFrame(
                {
                    "lrs": [str(lrs)],
                    "epochs": [str(epochs)],
                    "optims": [str(optims)],
                    "alpha": [str(alpha)],
                    "loss": [str(min_lose)],
                }
            )

            out.to_csv(output_path + "_e" + str(epoch) + ".csv")

    #
    #
    #
    #
    #

    tr_res = pd.DataFrame(
        {
            "train_info": train_info,
            "val_info": val_info,
        }
    )
    tr_res.to_csv("./saved_models/loss_" + now_time + ".csv")

    # Plot loss graph
    epoch_axis = np.arange(0, object_epoch)

    # plt.title("ACCURACY")

    # plt.plot(
    #     epoch_axis,
    #     [info["loss"] for info in train_info],
    #     epoch_axis,
    #     [info["loss"] for info in val_info],
    #     "r-",
    # )
    # plt.legend(["TRAIN", "VALIDATION"])

    # plt.figure()

    plt.title("LOSS")
    plt.plot(
        epoch_axis,
        [float(info) for info in train_info],
        epoch_axis,
        [float(info) for info in val_info],
        "r-",
    )
    plt.legend(["TRAIN", "VALIDATION"])

    plt.savefig("./saved_models/result_" + now_time + ".png")

    #
    #
    #
    #
    #

    model.cpu()
    model = ""
    torch.cuda.empty_cache()


optimss = [
    # optim.NAdam,
    optim.Adam,
]
# lrss = [0.00025]
# epochss = [130]
# alpha = [0.84]
lrss = [0.00003]
epochss = [200]
alpha = [0.84]

for o in optimss:
    for l in lrss:
        for e in epochss:
            for a in alpha:
                case_msg = (
                    "case: opt: "
                    + str(o)
                    + " / lr: "
                    + str(l)
                    + " / e: "
                    + str(e)
                    + " / a: "
                    + str(a)
                )
                print(case_msg)
                send_log(log_name, case_msg)
                main(l, e, o, a)



# main(0.0001, 1, optim.NAdam, 0.84)


#
#
#
#
#
#
#
#
#
#
# 사진 하나씩 불러오는 코드


def load_1_picture():

    prev_img = []

    for i, data in enumerate(tqdm.tqdm(train_dataloader)):
        if use_cuda:
            l = data["l"].to("cuda")
            ab = data["ab"].to("cuda")
            hint = data["hint"].to("cuda")
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

        # if len(prev_img) == 0:
        #     prev_img = gt_image


        # Loss func.

        # ssim_loss_val = ssim_loss(gt_image, hint_image)

        # l1_loss_val = l1_loss(gt_image, hint_image)

        # print("ssim_loss_val", ssim_loss_val)
        # print("l1_loss", l1_loss_val)

        # a = 0.84
        # L_mix = a * L_ms-ssim + (1-a) * L1 * Gaussian_L1

        ms_ssim_l1_loss = MS_SSIM_L1_LOSS(alpha=0.9, data_range=255)

        loss = ms_ssim_l1_loss(gt_image, hint_image)

        
        wandb.log({"MS_SSIM_L1_0.9": float(loss.cpu().flatten()[0])})


        # epoch (training / val)

        # test code

        # plt.figure(1)
        # plt.imshow(gt_bgr)
        # print(gt_bgr.shape)
        # plt.figure(2)
        # plt.imshow(hint_bgr)
        # print(hint_bgr.shape)
        # plt.show()

        # input()

        # prev_img = gt_image


# load_1_picture()