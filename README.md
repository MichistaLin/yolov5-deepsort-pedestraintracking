## 基于yolov5和deepsort的行人跟踪计数系统

项目运行环境：win10，pycharm，python3.6+

主要需要的包：pytorch >= 1.7.0，opencv

运行main.py即可开始追踪检测，可以在控制台运行

```python
python main.py --input="你的视频路径"
```

也可以在pycharm中直接右键运行（把--input中的defalt改为你要检测的视频路径即可），这样执行的都是默认参数

输入的参数：

```python
parser = argparse.ArgumentParser()
# 视频的路径，默认是本项目中的一个测试视频test.mp4，可自行更改
parser.add_argument('--input', type=str, default="./test.mp4",
                        help='test imgs folder or video or camera')  # 输入'0'表示调用电脑默认摄像头
# 处理后视频的输出路径
parser.add_argument('--output', type=str, default="./output",
                        help='folder to save result imgs, can not use input folder')
parser.add_argument('--weights', type=str, default='weights/yolov5l.pt', help='model.pt path(s)')
parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf_thres', type=float, default=0.6, help='object confidence threshold')
parser.add_argument('--iou_thres', type=float, default=0.4, help='IOU threshold for NMS')
# GPU（0表示设备的默认的显卡）或CPU
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# 通过classes来过滤检测类别
parser.add_argument('--classes', default=0, type=int, help='filter by class: --class 0, or --class 0 2 3')  

```

看到有挺多小伙伴要求总人数，现在加了一个参数`isCounntPresent`

```python
"""
  isCountPresent:
    True：表示只显示当前人数
    False：表示显示总人数和当前人数
"""
result_img = Counting_Processing(img, yolo5_config, Model, class_names, deepsort_tracker, Obj_Counter, isCountPresent = False)

```





检测效果：

![](https://img-blog.csdnimg.cn/965128beb6804047980329b7c4911275.jpeg#pic_center)