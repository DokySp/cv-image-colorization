import imp
from data.dataset import ColorHintDataset
import torch
import torch.utils.data as data
import cv2
import tqdm
from utils.logger import send_log
import torch
import os
import matplotlib.pyplot as plt
from utils.tensor2im import tensor2im

from models.model import AttentionR2Unet


root_path = "./datasets"
save_path = "./saved_models"
use_cuda = True

test_dataset = ColorHintDataset(root_path, 256, "test")
test_dataloader = data.DataLoader(test_dataset, batch_size=4, shuffle=False)


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


# TEST CODE

# TODO
input_path = "[ENTER_FILE_NANE]"
input_path = input_path.replace(" ", "__")
input_path = input_path.replace("-", "_")

model_path = os.path.join(save_path, input_path)  # basic_model.tar
saved_model = torch.load(model_path)


print(saved_model["accuracy"])


# TODO
# 모델을 불러오기
model = AttentionR2Unet().cuda()
model.load_state_dict(saved_model["state_dict"], strict=True)


# state_dict -> training한 모든 값들
# print(saved_model['state_dict'])


# TODO
# print(saved_model["state_dict"].keys())


# strict True -> 로드하는 모델 키와 적용하려는 모델 키가 같아야 됨!
#        False -> 이름 다른 경우는 버림
# result_path
