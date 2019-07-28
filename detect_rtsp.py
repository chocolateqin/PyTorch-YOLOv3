from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import cv2
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str,
                        default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str,
                        default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str,
                        default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float,
                        default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4,
                        help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int,
                        default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416,
                        help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str,
                        help="path to checkpoint model")
    parser.add_argument("--rtsp_url", type=str,
                        default="rtsp://192.168.67.116/live0.264", help="RTSP URL of a camera")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    # dataloader = DataLoader(
    #     ImageFolder(opt.image_folder, img_size=opt.img_size),
    #     batch_size=opt.batch_size,
    #     shuffle=False,
    #     num_workers=opt.n_cpu,
    # )
    camera = cv2.VideoCapture(opt.rtsp_url)

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print("\nPerforming object detection:")
    prev_time = time.time()
    # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    index = 0
    # plt.figure()
    # fig, ax = plt.subplots(1)
    while True:
        # Configure input
        res, frame = camera.read()

        if not res:
            break
        img_resized = cv2.resize(frame, (opt.img_size, opt.img_size))

        img = torch.from_numpy(img_resized.transpose(
            2, 0, 1)).float().div(255).unsqueeze(0).to(device)
        # print(img_tensor.size())
        # Get detections
        with torch.no_grad():
            detections = model(img)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Index %d, Inference Time: %s" % (index, inference_time))

        # Draw bounding boxes and labels of detections
        detection = detections[0]
        if detection is not None:
            # print(detection)
            # Rescale boxes to original image
            print(img_resized.shape[:2])
            detection = rescale_boxes(
                detection, opt.img_size, img_resized.shape[:2])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                # if cls_conf.item() <= 0.7:
                #     continue
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                print(x1.item(), y1.item(), x2.item(), y2.item())
                cv2.rectangle(img_resized, (int(x1.item()), int(y1.item())), (int(x2.item()), int(y2.item())), (0, 255, 0), 3)
        cv2.imshow("detection", img_resized)
        index += 1
        if cv2.waitKey(40) & 0xff == 27:
            break

    camera.release()
