import cv2
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

old_path = os.path.dirname(os.path.abspath(__file__))
read_path = "/Users/nishant/DataSet/train/sitting"
save_path = "/Users/nishant/DataSet/train/test_img"

def cnvt(img, name) :
    os.chdir(read_path)
    image = cv2.imread(img)

    os.chdir(old_path)
    model = 'mobilenet_thin'
    resolution = '432x368'
    w, h = model_wh(resolution)

    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    humans = e.inference(image)
    blank_image = np.zeros((h,w,3), np.uint8)
    image = TfPoseEstimator.draw_humans(blank_image, humans, imgcopy=False)

    os.chdir(save_path)
    cv2.imwrite(name, image)
    print("Saved - %s As - %s" % (img, name))

    os.chdir(old_path)


if __name__ == '__main__':
    os.chdir(old_path)
    pics = []

    os.chdir(read_path)
    for file in os.listdir("."):
        if file.endswith(".jpg"):
            pics.append(file)

    pics = sorted(pics)
    os.chdir(old_path)

    # i = 0
    # name = 'sitting'
    print("\nSTARTED\n")
    for pic in pics :
        name = 'sitting_'+pic[:-4]+'.jpg'
        cnvt(pic, name)
        # i += 1

    print("\nALL DONE\n")
