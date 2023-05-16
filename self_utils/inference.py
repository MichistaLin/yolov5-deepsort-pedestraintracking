import torch
import numpy as np
from utils.datasets import letterbox
from utils.utils import non_max_suppression


def img_preprocessing(np_img,device,newsize=640):
    np_img=letterbox(np_img,new_shape=newsize)[0]
    np_img = np_img[:, :, ::-1].transpose(2, 0, 1)
    np_img = np.ascontiguousarray(np_img)
    if device != "cpu":
        tensor_img=torch.from_numpy(np_img).to("cuda:{}".format(device))
    else:
        tensor_img = torch.from_numpy(np_img)
    tensor_img=tensor_img[np.newaxis,:].float()
    tensor_img /= 255.0
    return tensor_img


def yolov5_prediction(model,tensor_img,conf_thres,iou_thres,classes):
    # print(classes)
    with torch.no_grad():
        out=model(tensor_img)[0]
        pred = non_max_suppression(out, conf_thres, iou_thres, classes=classes)[0]
    return pred
