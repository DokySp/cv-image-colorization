from data.dataset import ColorHintDataset

import torch
import torch.utils.data as data
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

from utils.logger import send_log

log_name = "cv220525"


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

val_dataset = ColorHintDataset(root_path, 256, "val")
val_dataloader = data.DataLoader(val_dataset, batch_size=4, shuffle=False)

test_dataset = ColorHintDataset(root_path, 256, "test")
test_dataloader = data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# import ssim
# import torch.nn.functional as F

# ssim_loss = ssim.SSIM(mul=1000)
# l1_loss = torch.nn.L1Loss()

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


def test_1_epoch(model, dataloader, name):

    model.eval()  # PyTorch: train, test mode 운영

    # results = []

    for data in tqdm.auto.tqdm(dataloader):
        if use_cuda:
            l = data["l"].to("cuda")
            hint = data["hint"].to("cuda")
            file_name = data["file_name"]
        else:
            l = data["l"]
            hint = data["hint"]
            file_name = data["file_name"]

        hint_image = torch.cat((l, hint), dim=1)

        output = model(hint_image).squeeze()

        # batch size
        for i in range(4):
            output_np = tensor2im(output[i].unsqueeze(0))
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_LAB2RGB)
            cv2.imwrite("./result/" + name + "___" + file_name[i], output_bgr)
        
        l.to("cpu")
        hint.to("cpu")
        l = ""
        hint = ""

    # return results

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
    model = AttentionR2Unet().cuda()

    # Loss func.
    criterion = MS_SSIM_L1_LOSS(alpha=alpha)

    # 옵티마이저

    optimizer = optims(model.parameters(), lr=lrs)  # 학습할 것들을 옵팀에 넘김

    # 기타 변수들
    train_info = []
    val_info = []
    object_epoch = epochs


    save_path = "./saved_models"
    os.makedirs(save_path, exist_ok=True)
    # output_path = os.path.join(save_path, "basic_model.tar") # 관습적으로 tar을 확장자로 사용
    # output_path = os.path.join(save_path, "validation_model.tar")
    output_path = os.path.join(save_path, "AttentionR2Unet_" + now_time + ".pth")
    output_path = output_path.replace(" ", "__")
    output_path = output_path.replace("-", "_")



    #
    #
    #
    #
    #


    min_lose = 100000000000000

    for epoch in range(object_epoch):
        train_loss = train_1_epoch(model, train_dataloader, optimizer=optimizer, criterion=criterion)
        test_val = "[Training] Epoch {}: loss: {}".format(
                epoch, train_loss
            )
        
        print(test_val)
        send_log(log_name, test_val)
    
        train_loss = train_loss.detach().cpu().flatten()[0]
        train_info.append(train_loss)

        # Validation
        with torch.no_grad():  # gradient 계산 X
            val_loss = validation_1_epoch(model, val_dataloader, criterion=criterion)
        
        val_msg = "[Validation] Epoch: {}, loss: {}".format(
                epoch, val_loss
            )
        
        print(val_msg)
        send_log(log_name, val_msg)

        val_loss = val_loss.detach().cpu().flatten()[0]
        val_info.append(val_loss)

        # 제일 정확한 모델만 저장!
        if min_lose > val_loss:
            min_lose = val_loss
            min_loss_msg = "min_loss: " + str(min_lose)
            print(min_loss_msg)
            send_log(log_name, min_loss_msg)

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
                output_path,
            )

            out = pd.DataFrame({
                "lrs": [str(lrs)], 
                "epochs": [str(epochs)], 
                "optims": [str(optims)],
                "alpha": [str(alpha)],
                "loss": [str(min_lose.detach().cpu().numpy())],
            })

            out.to_csv(output_path+".csv")


    #
    #
    #
    #
    #

    tr_res = pd.DataFrame({
        "train_info": train_info,
        "val_info": val_info,
    })
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
    optim.NAdam,
    # optim.Adam,
]
lrss = [0.00025]
# epochss = [130]
epochss = [1]
alpha = [0.84]

for o in optimss:
    for l in lrss:
        for e in epochss:
            for a in alpha:
                case_msg = "case: opt: " + str(o) + " / lr: " + str(l) + " / e: " + str(e) + " / a: " + str(a)
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
