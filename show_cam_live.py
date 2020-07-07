import cv2
import tensorflow as tf
import numpy as np
import os
#import pandas as pd
from threading import Thread
import queue
import threading
from predict_cards import predict_box, show_class, draw_bounding_box, predictThread # , predict_image
from predict_cards_RetNet import draw_detections, predict_image #, predictThread

# For using RetinaNet bounding box detector:
# - import predictThread from predict_cards_RetNet
# - comment out: img = draw_bounding_box(img, bboxes)
# - comment out: img = show_class(img, classes, bboxes)
# - comment in: draw_detections(img, bboxes, classes)

#from goprocam import GoProCamera
#from goprocam import constants

# cascPath = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)
# gpCam = GoProCamera.GoPro()
# gpCam.stream('udp://127.0.0.1:10000')
# cap = cv2.VideoCapture("udp://127.0.0.1:10000")

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)   # 0 for build in cam, 1 for usb cam
    if not cam.isOpened():
        print('Could not open video')
        return
    frame = 200
    pred_thread = predictThread()
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)

        if frame > 100:
            t = Thread(target=pred_thread.run, args=(img,))
            t.start()
            print("new prediction")
            frame = 0
        bboxes = pred_thread.boxes
        classes = pred_thread.labels
        img = draw_bounding_box(img, bboxes)
        img = show_class(img, classes, bboxes)
        #draw_detections(img, bboxes, classes)
        frame += 1

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=False)
    #predict_image('test.png')


if __name__ == '__main__':
    main()