import cv2
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

THRESH_SCORE = 0.6
classifier_model = tf.keras.models.load_model('models/CardClassifier.h5')
bbox_model = tf.keras.models.load_model('models/BoundingBox_Locator.h5',
                                        custom_objects={'leaky_relu': tf.nn.leaky_relu})
model = models.load_model('models/resnet50_csv_10.h5', backbone_name='resnet50')
model = models.convert_model(model)
labels_to_names = pd.read_csv(
  'classes.csv',
  header=None,
  index_col=0
).to_dict()[1]


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(1)   # 0 for build in cam, 1 for usb cam
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        bboxes, scores, classes = predict_box_resnet(img)
        img = draw_bounding_box(img, bboxes)
        img = show_class(img, classes, bboxes)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def predict_labels(img, boxes):
    img = img[0]
    labels = []
    for i, box in enumerate(boxes[0]):
        crop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        crop = cv2.resize(crop, (80, 80), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        crop = cv2.Canny(np.uint8(gray), 30, 200)
        crop = crop/255.0
        crop = crop.reshape(-1, 80, 80, 1)
        prediction = classifier_model.predict(crop)
        idx = np.argmax(prediction[0])
        for key, value in labels_to_names.items():
            if value == idx:
                labels.append(key)
                break
    return labels


def predict_box(image):
    image = cv2.resize(image, (135, 135))
    image = np.expand_dims(image, axis=0)
    boxes = bbox_model.predict(image)

    labels = predict_labels(image, boxes)
    #boxes /= scale
    return boxes, labels


THRES_SCORE = 0.6
def predict_labels_resnet(img, boxes, scores):
  img = img[0]
  labels = []
  for i, box in enumerate(boxes[0]):
    if scores[0][i] < THRES_SCORE:
      break
    crop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    crop = cv2.resize(crop, (80, 80), interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    crop = cv2.Canny(np.uint8(gray), 30, 200)
    crop = crop/255.0
    crop = crop.reshape(-1, 80, 80, 1)
    prediction = classifier_model.predict(crop)
    idx = np.argmax(prediction[0])
    for key, value in labels_to_names.items():
      if value == idx:
        labels.append(key)
        break

  return labels


def predict_box_resnet(image):
    image = preprocess_image(image.copy())
    image, scale = resize_image(image)
    image = np.expand_dims(image, axis=0)
    boxes, scores, labels = model.predict_on_batch(
        image
    )

    labels = predict_labels_resnet(image, boxes, scores)
    boxes /= scale
    return boxes, scores, labels


def show_class(image, cards, bboxes):
    for i in range(len(cards)):
        cv2.putText(image, cards[i], (round(bboxes[0][i][0]), round(bboxes[0][i][1])), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 255), 2)
    return image


def draw_bounding_box(image, bboxes):
    for bbox in bboxes[0]:
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return image


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()