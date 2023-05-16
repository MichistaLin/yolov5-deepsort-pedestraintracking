import torch, sys, argparse, cv2, os, time
from datetime import datetime
from self_utils.multi_tasks import Counting_Processing
from self_utils.overall_method import Object_Counter, Image_Capture
from deep_sort.configs.parser import get_config
from deep_sort.deep_sort import DeepSort
import imutils


def main(yolo5_config):
    print("=> main task started: {}".format(datetime.now().strftime('%H:%M:%S')))
    a = time.time()
    class_names = []
    # 加载模型
    if yolo5_config.device != "cpu":
        Model = torch.load(yolo5_config.weights, map_location=lambda storage, loc: storage.cuda(int(yolo5_config.device)))[
        'model'].float().fuse().eval()
    else:
        Model = torch.load(yolo5_config.weights, map_location=torch.device('cpu'))['model'].float().fuse().eval()
    # 模型能检测的类别['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', ...]
    classnames = Model.module.names if hasattr(Model, 'module') else Model.names
    # print(classnames)
    # 只检测人这一个类别，所以只需要在框上面标识出是person即可
    # 如果想检测其他物体需要在100行附近更改'--classes'的数值，然后在这里把标签改为对应即可
    class_names.append(classnames[0])
    b = time.time()
    print("==> class names: ", class_names)
    print("=> load model, cost:{:.2f}s".format(b - a))

    os.makedirs(yolo5_config.output, exist_ok=True)
    c = time.time()
    # 初始化追踪器deepssort_tracker
    cfg = get_config()
    cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
    deepsort_tracker = DeepSort(cfg.DEEPSORT.REID_CKPT, max_dist=cfg.DEEPSORT.MAX_DIST,
                                min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, max_age=cfg.DEEPSORT.MAX_AGE,
                                n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                use_cuda=True, use_appearence=True)

    # 输入需要检测的图片或图片目录或视频并处理
    mycap = Image_Capture(yolo5_config.input)
    # 实例化计数器
    Obj_Counter = Object_Counter(class_names)
    # 总帧数
    total_num = mycap.get_length()
    videowriter = None
    fps = int(mycap.get(5))
    t = int(1000 / fps)
    while mycap.ifcontinue():
        ret, img = mycap.read()
        if ret:
            # 开始检测图片中的人
            result_img = Counting_Processing(img, yolo5_config, Model, class_names, deepsort_tracker, Obj_Counter)
            # print(result_img)
            if videowriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videowriter = cv2.VideoWriter(
                    './output/result.mp4', fourcc, fps, (result_img.shape[1], result_img.shape[0]))
            videowriter.write(result_img)
            result_img = imutils.resize(result_img, height=500)
            cv2.imshow('video', result_img)
            cv2.waitKey(t)

            if cv2.getWindowProperty('video', cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

        sys.stdout.write("\r=> processing at %d; total: %d" % (mycap.get_index(), total_num))
        sys.stdout.flush()

    videowriter.release()
    cv2.destroyAllWindows()
    mycap.release()
    print(
        "\n=> process done {}/{} images, total cost: {:.2f}s [{:.2f} fps]".format(len(os.listdir(yolo5_config.output)),
                                                                                  total_num, time.time() - c,
                                                                                  len(os.listdir(
                                                                                      yolo5_config.output)) / (
                                                                                              time.time() - c)))

    print("=> main task finished: {}".format(datetime.now().strftime('%H:%M:%S')))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
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
    # 通过classes来过滤yolo要检测类别，0表示检测人，1表示自行车，更多具体类别数字可以在19行附近打印出来
    parser.add_argument('--classes', default=0, type=int, help='filter by class: --class 0, or --class 0 1 2 3')

    yolo5_config = parser.parse_args()
    print(yolo5_config)
    main(yolo5_config)
    print("结果保存在：", yolo5_config.output)
