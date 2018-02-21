import argparse
import logging
import time

import cv2
import numpy as np

import os

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

old_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.dirname(os.path.abspath(__file__))+"/new_frames"
read_path = "/Users/nishant/DataSet/train/sitting"

def cnvt(img) :
    os.chdir(read_path)
    image = cv2.imread(img)
    humans = e.inference(image)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    os.chdir(save_path)
    cv2.imwrite(img, image)
    os.chdir(old_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    # parser.add_argument('--image', type=str, default='')
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()


    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    os.chdir(read_path)
    pics = []

    for file in os.listdir("."):
        if file.endswith(".jpg"):
            pics.append(file)

    pics = sorted(pics)
    os.chdir(old_path)

    for pic in pics :
        cnvt(pic)

    """
    image = cv2.imread(args.image)
    humans = e.inference(image)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    cv2.imshow('tf-pose-estimation result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('RESULT.jpg', image)
    """

logger.debug('finished+')
