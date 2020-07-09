import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import threading

THRESH_SCORE = 0.6
classifier_model = tf.keras.models.load_model('models/CardClassifier.h5')
bbox_model = tf.keras.models.load_model('models/BoundingBox_Locator.h5',
                                        custom_objects={'leaky_relu': tf.nn.leaky_relu})

labels_to_names = pd.read_csv(
  'classes.csv',
  header=None,
  index_col=0
).to_dict()[1]

scale_factor = 0


def predict_labels(img, boxes):
    img = img[0]
    labels = []
    for i, box in enumerate(boxes):
        if any(x < 10 for x in box[0]):
            continue
        #print(box[0])
        crop = img[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])]
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

def show_class(image, cards, bboxes):
    for i in range(len(cards)):
        cv2.putText(image, cards[i], (round(bboxes[i][0][0]), round(bboxes[i][0][1])), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 255), 2)
    return image


def draw_bounding_box(image, bboxes):
    for bbox in bboxes:
        image = cv2.rectangle(image, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3]), (255, 0, 0), 2)
    return image


def predict_box(image):
    global scale_factor
    orig_shape = np.shape(image)
    image = cv2.resize(image, (135, 135))
    resized_image_shape = np.shape(image)
    scale_factor = np.flip(np.divide(orig_shape, resized_image_shape))
    image = np.expand_dims(image, axis=0)
    boxes = bbox_model.predict(image)

    labels = predict_labels(image, boxes)
    #boxes /= scale
    for box in boxes:
        box[0][0] = box[0][0] * scale_factor[1]
        box[0][1] = box[0][1] * scale_factor[2]
        box[0][2] = box[0][2] * scale_factor[1]
        box[0][3] = box[0][3] * scale_factor[2]
    return boxes, labels


class predictThread():
   def __init__(self):
        self.boxes = []
        self.labels = []

   def run(self, img):
        self.boxes, self.labels = predict_box(img)



def predict_image(filepath):
    global scale_factor
    img = cv2.imread(filepath)
    boxes, classes = predict_box(img)
    img = show_class(img, classes, boxes)
    img = draw_bounding_box(img, boxes)
    #img = cv2.rectangle(img, (50, 320), (92, 333), (255, 0, 0), -1)
    print(boxes)
    print(classes)
    cv2.imshow('image', img)
    cv2.waitKey()