# cringe
import cv2
import numpy as np

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from random import shuffle
from tqdm import tqdm

old_path = os.path.dirname(os.path.abspath(__file__))
# old_path = os.getcwd()
MODEL = 'mobilenet_thin'
resolution = '432x368'
w, h = model_wh(resolution)

# e = TfPoseEstimator(get_graph_path(MODEL), target_size=(w, h))
# humans = e.inference(image)
# blank_image = np.zeros((h,w,3), np.uint8)
# image = TfPoseEstimator.draw_humans(blank_image, humans, imgcopy=False)

TRAIN_DIR = '/Users/nishant/DataSet/train/test_img'
TEST_DIR = '/Users/nishant/DataSet/test'

LR=1e-3

def label_img(img):
    word_label = img.split('_')[0]

    if word_label == 'sitting':
        return [1,0,0,0]
    elif word_label == 'standing':
        return [0,1,0,0]
    else:
        return [0,0,1,0]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        # img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    # print("\n\n%s\n\n" % training_data)
    np.save('train_data.npy', training_data)
    return training_data

train_data = create_train_data()

# import tensorflow as tf
import tflearn

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None,w,h,1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet,3, activation='linear')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


os.chdir('..')
os.chdir('./models/graph/mobilenet_thin/')
# cmu = './models/graph/cmu/graph_opt.pb'
# mobilenet_thin = './models/graph/mobilenet_thin/graph_opt.pb'

if os.path.exists('{}.pb'. format(MODEL)):
    model.load(MODEL)
    print('model loaded!')
os.chdir(old_path)


train = train_data[:-5]
test = train_data[-5:]


X = np.array([i[0] for i in train]).reshape(-1,w,h,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,w,h,1)
test_y = [i[1] for i in test]


model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=20, show_metric=True, run_id=MODEL)
