import cv2
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from goprocam import GoProCamera
from goprocam import constants

# cascPath = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)
# gpCam = GoProCamera.GoPro()
# gpCam.stream('udp://127.0.0.1:10000')
# cap = cv2.VideoCapture("udp://127.0.0.1:10000")
THRESH_SCORE = 0.6
classifier_model = tf.keras.models.load_model('models/CardClassifier.h5')
bbox_model = tf.keras.models.load_model('models/BoundingBox_Locator.h5',
                                        custom_objects={'leaky_relu': tf.nn.leaky_relu})
scale_factor = 0
labels_to_names = pd.read_csv(
  'classes.csv',
  header=None,
  index_col=0
).to_dict()[1]


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(1)   # 0 for build in cam, 1 for usb cam
    if not cam.isOpened():
        print('Could not open video')
        return
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        # bboxes, classes = predict_box(img)
        # for box in bboxes:
        #     box[0][0] = box[0][0] * scale_factor[1]
        #     box[0][1] = box[0][1] * scale_factor[2]
        #     box[0][2] = box[0][2] * scale_factor[1]
        #     box[0][3] = box[0][3] * scale_factor[2]
        # img = draw_bounding_box(img, bboxes)
        # img = show_class(img, classes, bboxes)
        # print(np.shape(img))

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


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


# def predict_box_resnet(image):
#     image = preprocess_image(image.copy())
#     image, scale = resize_image(image)
#     image = np.expand_dims(image, axis=0)
#     #boxes, scores, labels = model.predict_on_batch(image)
#
#     labels = predict_labels_resnet(image, boxes, scores)
#     boxes /= scale
#     return boxes, scores, labels


def show_class(image, cards, bboxes):
    for i in range(len(cards)):
        cv2.putText(image, cards[i], (round(bboxes[i][0][0]), round(bboxes[i][0][1])), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 255), 2)
    return image


def draw_bounding_box(image, bboxes):
    for bbox in bboxes:
        image = cv2.rectangle(image, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3]), (255, 0, 0), 2)
    return image


def predict_image(filepath):
    global scale_factor
    img = cv2.imread(filepath)
    boxes, classes = predict_box(img)
    print(scale_factor)
    for box in boxes:
        box[0][0] = box[0][0]*scale_factor[1]
        box[0][1] = box[0][1]*scale_factor[2]
        box[0][2] = box[0][2]*scale_factor[1]
        box[0][3] = box[0][3]*scale_factor[2]
    img = show_class(img, classes, boxes)
    img = draw_bounding_box(img, boxes)
    #img = cv2.rectangle(img, (50, 320), (92, 333), (255, 0, 0), -1)
    print(boxes)
    print(classes)
    cv2.imshow('image', img)
    cv2.waitKey()


def main():
    show_webcam(mirror=False)
    #predict_image('test.png')


if __name__ == '__main__':
    main()