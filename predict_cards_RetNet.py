import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import threading
import sys
import matplotlib.pyplot as plt

#sys.path.append('/keras-retinanet-master/')
from keras_retinanet_master.keras_retinanet import models as retmodels
from keras_retinanet_master.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet_master.keras_retinanet.utils.visualization import draw_box, draw_caption

THRESH_SCORE = 0.6
classifier_model = tf.keras.models.load_model('models/CardClassifier_merged_dataset.h5')

pretrained_path = 'models/resnet50_own_dataset_20.h5'
RetNet_model = retmodels.load_model(pretrained_path, backbone_name='resnet50')
RetNet_model = retmodels.convert_model(RetNet_model)

labels_to_names = pd.read_csv(
  'classes.csv',
  header=None,
  index_col=0
).to_dict()[1]

scale_factor = 0

THRES_SCORE = 0.6
def draw_detections(image, boxes, labels):
    if boxes == []:
        return
    for box, label in zip(boxes, labels):
        color = [0, 255, 0]
        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{}".format(label)
        draw_caption(image, b, caption)


def predict_labels_resnet(img, boxes, scores):
    img = img[0]
    labels = []
    confis = []
    for i, box in enumerate(boxes[0]):
        if scores[0][i] < THRES_SCORE:
            break
        crop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        crop = cv2.resize(crop, (80, 80), interpolation = cv2.INTER_AREA)
        cv2.imwrite("test_rgb.png", crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.Canny(np.uint8(gray), 10, 100)
        cv2.imwrite("test3.png", crop)
        crop = crop/255.0 
        crop = crop.reshape(-1, 80, 80, 1)
        prediction = classifier_model.predict(crop)

        idx = np.argmax(prediction[0])
        confis.append(prediction[0][idx])
        for key, value in labels_to_names.items():
            if value == idx:
                labels.append(key)
                break
    boxes = boxes[0,:len(labels),:]
    return boxes, labels#, confis


def predict_box_resnet(image):
    image = preprocess_image(image.copy())
    image, scale = resize_image(image)
    image = np.expand_dims(image, axis=0)
    boxes, scores, labels = RetNet_model.predict_on_batch(image)

    boxes, labels = predict_labels_resnet(image, boxes, scores)
    boxes /= scale
    return boxes, labels


class predictThread():
   def __init__(self):
        self.boxes = []
        self.labels = []

   def run(self, img):
        self.boxes, self.labels = predict_box_resnet(img)



def predict_image(filepath):
    global scale_factor
    img = cv2.imread(filepath)
    boxes, classes = predict_box_resnet(img)
    draw = img.copy()
    draw_detections(draw, boxes, classes)
    print(boxes)
    print(classes)
    cv2.imshow('image', draw)
    cv2.waitKey()

def predict_cropped():
    crop = cv2.imread("test2.png")
    crop = crop.reshape(-1, 80, 80, 1)
    prediction = classifier_model.predict(crop)
    idx = np.argmax(prediction[0])
    print(idx)
    for key, value in labels_to_names.items():
        if value == idx:
            print(key)
            break