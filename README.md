## 基于yolov5和deepsort的行人跟踪计数系统

项目运行环境：win10，pycharm，python3.6+

主要需要的包：pytorch >= 1.7.0，opencv

运行main.py即可开始追踪检测，可以在控制台运行也可以在pycharm中直接右键运行（把--input中的defalt改为你要检测的视频路径即可

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
parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou_thres', type=float, default=0.4, help='IOU threshold for NMS')
# GPU（0表示设备的默认的显卡）或CPU
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--classes', default=0, type=int, help='filter by class: --class 0, or --class 0 2 3')  # 通过classes来过滤检测类别

```



检测效果：

![](.\output\demo.jpg)