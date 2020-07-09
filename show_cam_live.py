import cv2
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from threading import Thread
import queue
import threading
#from predict_cards import predict_box, show_class, draw_bounding_box#, predictThread # , predict_image
from predict_cards_RetNet import draw_detections, predict_image, predictThread, predict_cropped
import game_prediction


THRESH_SCORE = 0.6

num_players = 1
scale_factor = 0
labels_to_names = pd.read_csv(
  'classes.csv',
  header=None,
  index_col=0
).to_dict()[1]
predictor = game_prediction.Predictor()

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(1)   # 0 for build in cam, 1 for usb cam
    if not cam.isOpened():
        print('Could not open video')
        return
    pred_thread = predictThread()
    bboxes = []
    classes = []
    old_classes = []
    frame = 100
    win_chance = -1
    loose_chance = -1
    position = []
    WIDTH = 640
    HEIGHT = 480
    while True:
        ret_val, img = cam.read()
        #print(img.shape)

        if mirror:
            img = cv2.flip(img, 1)

        if frame > 60:
            t = Thread(target=pred_thread.run, args=(img,))
            t.start()
            #dprint("new prediction")
            frame = 0
        bboxes = pred_thread.boxes
        classes = pred_thread.labels
        if not old_classes == classes:
            print("Update")
            predictor.update(bboxes, classes)
            for i in range(num_players):
                win_chance, loose_chance, position = predictor.predict_winning_loosing(i)
            old_classes = classes
        if win_chance > -1:
            img = show_chance(img, position, win_chance, loose_chance)
        #img = draw_bounding_box(img, bboxes)
        #img = show_class(img, classes, bboxes)
        draw_detections(img, bboxes, classes)
        img = cv2.line(img, (320,0), (320,480), (255,0,0), thickness=2)
        img = cv2.line(img, (0,240), (640,240), (255,0,0), thickness=2)
        frame += 1

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()






def show_chance(image, position, win, loose):
    cv2.putText(image, str(win), (round(position[0]), round(position[1])), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 2)  # green color
    cv2.putText(image, str(loose), (round(position[2]), round(position[3])), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 255), 2)  # red color
    return image


def show_class(image, cards, bboxes):
    for i in range(len(cards)):
        cv2.putText(image, cards[i], (round(bboxes[i][0][0]), round(bboxes[i][0][1])), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 255), 2)
    return image


def draw_bounding_box(image, bboxes):
    for bbox in bboxes:
        image = cv2.rectangle(image, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3]), (255, 0, 0), 2)
    return image


def main():
    show_webcam(mirror=False)
    #predict_image('test.png')
    #predict_cropped()


if __name__ == '__main__':
    main()