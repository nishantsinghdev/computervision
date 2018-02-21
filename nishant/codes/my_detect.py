import cv2
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

import tensorflow as tf

if __name__ == '__main__':
    model = 'mobilenet_thin'
    # model = 'cmu'

    resolution = '432x368'
    w, h = model_wh(resolution)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)

        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        image = cv2.imread('test5.jpg')
        humans = e.inference(image)
        blank_image = np.zeros((h,w,3), np.uint8)
        image = TfPoseEstimator.draw_humans(blank_image, humans, imgcopy=False)

        cv2.imshow('tf-pose-estimation result', image)
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    writer.close()
    cv2.imwrite('RESULT.jpg', image)
