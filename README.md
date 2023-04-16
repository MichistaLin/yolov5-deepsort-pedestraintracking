## 基于yolov5和deepsort的行人跟踪计数系统

项目运行环境：win10，pycharm，python3.6+

主要需要的包：pytorch >= 1.7.0，opencv

考虑到pytorch如果想在显卡上跑，需要gpu版本的，并且各种库的版本更新较快，难免与旧的代码有冲突，所以下面给出了带有虚拟环境的项目的网盘链接（因为pytorch实在是太大了），能跑的话就不要更新里面的库。

<a href="https://www.123pan.com/s/XuubVv-x2mvd.html">基于yolov5和deepsort的行人跟踪计数系统</a>

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



检测效果：

![](.\output\demo.jpg)