import cv2
import numpy as np

from . inference import yolov5_prediction,img_preprocessing
from . post_processing import count_post_processing


def Counting_Processing(input_img,yolo5_config,model,class_names,Tracker,Obj_Counter, isCountPresent):
    try:
        # 把图片转换成tensor类型
        tensor_img=img_preprocessing(input_img,yolo5_config.device,yolo5_config.img_size)
        # 使用yolov5进行检测
        pred=yolov5_prediction(model,tensor_img,yolo5_config.conf_thres, yolo5_config.iou_thres,yolo5_config.classes)
        # 跟踪计数
        result_img=count_post_processing(input_img,pred,class_names,tensor_img.shape,Tracker,Obj_Counter, isCountPresent)

        return result_img

    except Exception as e:
        return e

    
def Background_Modeling(myP,input_img,save_path,bg_model):
    try:
        fg_mask = bg_model.apply(input_img)
        bg_img = bg_model.getBackgroundImage()
        cv2.putText(input_img,"origin image",(5,80),cv2.FONT_HERSHEY_TRIPLEX, 1.6, [0,200,0],thickness=3)
        cv2.putText(bg_img,"background image",(5,80),cv2.FONT_HERSHEY_TRIPLEX, 1.6, [0,200,0],thickness=3)
        result_img=np.vstack([input_img, bg_img])
        if myP is not None:
            myP.apply_async(cv2.imwrite,(save_path,result_img,))
        else:
            cv2.imwrite(save_path,result_img)
        return True,save_path
    except Exception as e:
        print("Wrong:",e,save_path)
        return False,e
    