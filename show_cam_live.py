import cv2
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from threading import Thread
import queue
import requests
import threading
#from predict_cards import predict_box, show_class, draw_bounding_box#, predictThread # , predict_image
from predict_cards_RetNet import draw_detections, predict_image, predictThread, predict_cropped
from game_manager import GameManager

labels_to_names = pd.read_csv(
  'classes.csv',
  header=None,
  index_col=0
).to_dict()[1]

WIDTH = 1280#1920
HEIGHT = 720#1080
Manager = GameManager(WIDTH, HEIGHT)

url = "http://192.168.1.19:8080/shot.jpg"


def show_webcam(mirror=False):
    #cam = cv2.VideoCapture(1)   # 0 for build in cam, 1 for usb cam
    #if not cam.isOpened():
    #    print('Could not open video')
    #    return
    pred_thread = predictThread()
    bboxes = []
    classes = []
    confis = []
    frame = 0
    cv2.namedWindow('webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('webcam', WIDTH, HEIGHT)
    while True:
        #ret_val, img = cam.read()
        #print(img.shape)
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        if mirror:
            img = cv2.flip(img, 1)

        if frame > 30:
            t = Thread(target=pred_thread.run, args=(img,))
            t.start()
            Manager.update(bboxes, classes, confis)
            #print("new prediction")
            frame = 0
        bboxes = pred_thread.boxes
        classes = pred_thread.labels
        confis = pred_thread.confis
        img = Manager.show_chance(img)
        draw_detections(img, bboxes, classes, confis)
        #img = cv2.line(img, (WIDTH//2,0), (WIDTH//2,HEIGHT), (255,0,0), thickness=2)
        #img = cv2.line(img, (0,HEIGHT//2), (WIDTH,HEIGHT//2), (255,0,0), thickness=2)
        frame += 1
        cv2.imshow('webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def show_chance(image, position, win, loose):
    cv2.putText(image, str(win), (round(position[0]), round(position[1])), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 2)  # green color
    cv2.putText(image, str(loose), (round(position[2]), round(position[3])), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 255), 2)  # red color
    return image


def main():
    show_webcam(mirror=False)
    #predict_image('15.jpg')
    #predict_cropped()


if __name__ == '__main__':
    main()