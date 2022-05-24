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

val_dataset = ColorHintDataset(root_path, 256, "val")
val_dataloader = data.DataLoader(val_dataset, batch_size=4, shuffle=False)

test_dataset = ColorHintDataset(root_path, 256, "test")
test_dataloader = data.DataLoader(val_dataset, batch_size=4, shuffle=False)

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


# Load Model
model = AttentionR2Unet().cuda()

# Loss func.
criterion = MS_SSIM_L1_LOSS(alpha=0.84)

# 옵티마이저
import torch.optim as optim

optimizer = optim.NAdam(model.parameters(), lr=0.0001)  # 학습할 것들을 옵팀에 넘김

# 기타 변수들
train_info = []
val_info = []
object_epoch = 10


save_path = "./saved_models"
os.makedirs(save_path, exist_ok=True)
# output_path = os.path.join(save_path, "basic_model.tar") # 관습적으로 tar을 확장자로 사용
# output_path = os.path.join(save_path, "validation_model.tar")
output_path = os.path.join(save_path, "AttentionR2Unet" + datetime.now() + ".tar")


def train_1_epoch(model, dataloader):
    # 시각화를 위한 변수
    # confusion_matrix = [[0, 0], [0, 0]]
    total_loss = 0
    iteration = 0

    model.train()  # PyTorch: train, test mode 운영

    for i, data in tqdm.auto.tqdm(dataloader):

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

        print(output)
        print(loss)
        print(i)
        input()

        # back propagation
        loss.backward()

        # Gradient의 learnable parameter update (lr, Adam 안에 기타 변수들 등등)
        optimizer.step()

        # ----------

        total_loss += loss.detach()  # detach -> parameter 연산에 사용 X
        iteration += 1

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
#
#
#
#
#
#
#
#
#


def validation_1_epoch(model, dataloader):
    # 시각화를 위한 변수
    # confusion_matrix = [[0, 0], [0, 0]]
    total_loss = 0
    iteration = 0

    model.eval()  # PyTorch: train, test mode 운영

    for i, data in tqdm.auto.tqdm(dataloader):

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

        print(loss)
        print(i)
        input()

        # for i in range(len(label)):
        #     real_class = int(label[i])
        #     pred_class = int(output[i] > 0.5)
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
#
#
#
#
#
#
#
#


best_accuracy = 0

for epoch in range(object_epoch):
    train_accuracy, train_loss = train_1_epoch(model, train_dataloader)
    print(
        "[Training] Epoch {}: accuracy: {}, loss: {}".format(
            epoch, train_accuracy, train_loss
        )
    )

    train_info.append({"loss": train_loss, "accuracy": train_accuracy})

    # Validation
    with torch.no_grad():  # gradient 계산 X
        val_accuracy, val_loss = validation_1_epoch(model, val_dataloader)
    print(
        "[Validation] Epoch {}: accuracy: {}, loss: {}".format(
            epoch, val_accuracy, val_loss
        )
    )

    val_info.append({"loss": val_loss, "accuracy": val_accuracy})

    # 제일 정확한 모델만 저장!
    if best_accuracy < val_accuracy:
        best_accuracy = val_accuracy

        torch.save(
            {
                "memo": "This is Classification Model",
                "accuracy": best_accuracy,
                "state_dict": model.state_dict(),  # 모든 weight 변수이름 / parameter 값들을 가진 dict.
            },
            output_path,
        )


#
#
#
#
#
#
#
#


import pandas as pd

tr_res = pd.DataFrame(train_info)
va_res = pd.DataFrame(val_info)

tr_res.to_csv("train" + datetime.now() + ".csv")
va_res.to_csv("validation" + datetime.now() + ".csv")


import numpy as np
import matplotlib.pyplot as plt

epoch_axis = np.arange(0, object_epoch)

plt.title("ACCURACY")

plt.plot(
    epoch_axis,
    [info["accuracy"] for info in train_info],
    epoch_axis,
    [info["accuracy"] for info in val_info],
    "r-",
)
plt.legend(["TRAIN", "VALIDATION"])

plt.figure()

plt.title("LOSS")
plt.plot(
    epoch_axis,
    [info["loss"].detach().cpu().numpy() for info in train_info],
    epoch_axis,
    [info["loss"].detach().cpu().numpy() for info in val_info],
    "r-",
)
plt.legend(["TRAIN", "VALIDATION"])

plt.savefig("result" + datetime.now() + ".png")


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


# inference step
def test_1epoch(model, dataloader):

    model.eval()  # PyTorch: train, test mode 운영

    results = []

    for i, data in tqdm.auto.tqdm(dataloader):
        if use_cuda:
            l = data["l"].to("cuda")
            hint = data["hint"].to("cuda")
            file_name = data["file_name"].to("cuda")
        else:
            l = data["l"]
            hint = data["hint"]
            file_name = data["file_name"]

        hint_image = torch.cat((l, hint), dim=1)

        output = model(hint_image).squeeze()

        output_np = tensor2im(output)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_LAB2RGB)
        cv2.imwrite("./result/" + file_name, output_bgr)

    return results


def test():
    model_path = os.path.join(save_path, "deep_model.tar")  # basic_model.tar
    saved_model = torch.load(model_path)

    print(saved_model["memo"])
    print(saved_model.keys())
    print(saved_model["accuracy"])

    # 모델을 불러오기
    model = AttentionR2Unet().cuda()
    model.load_state_dict(saved_model["state_dict"], strict=True)

    # state_dict -> training한 모든 값들
    # print(saved_model['state_dict'])
    print(saved_model["state_dict"].keys())
    # strict True -> 로드하는 모델 키와 적용하려는 모델 키가 같아야 됨!
    #        False -> 이름 다른 경우는 버림
    result_path = "./output.txt"

    res = test_1epoch(model, test_dataloader)
    with open(result_path, "w") as f:
        f.writelines(res)


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
