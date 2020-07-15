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
classifier_model = tf.keras.models.load_model('models/CardClassifier_sobel_drop.h5')

pretrained_path = 'models/resnet50_own_dataset_20.h5'
RetNet_model = retmodels.load_model(pretrained_path, backbone_name='resnet50')
RetNet_model = retmodels.convert_model(RetNet_model)

labels_to_names = pd.read_csv(
  'classes.csv',
  header=None,
  index_col=0
).to_dict()[1]


THRES_SCORE = 0.6
def draw_detections(image, boxes, labels):
    if boxes == []:
        return
    for box, label in zip(boxes, labels):
        color = [0, 255, 0]
        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{}".format(label)
        #draw_caption(image, b, caption)
        #b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

scale = 1
delta = 0
ddepth = cv2.CV_16S
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

        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
      
        crop = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        #crop = cv2.Canny(np.uint8(gray), 30, 90)
        cv2.imwrite("test2.png", crop)
        crop = crop/255.0 
        crop = crop.reshape(-1, 80, 80, 1)
        prediction = classifier_model.predict(crop)

        idx = np.argmax(prediction)%52
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
    #print(scale) # 1.6666666666666
    return boxes, labels


class predictThread():
   def __init__(self):
        self.boxes = []
        self.labels = []

   def run(self, img):
        self.boxes, self.labels = predict_box_resnet(img)



def predict_image(filepath):
    # Reading image 
    font = cv2.FONT_HERSHEY_COMPLEX 
    img2 = cv2.imread(filepath, cv2.IMREAD_COLOR) 
    
    # Reading same image in another  
    # variable and converting to gray scale. 
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) 

    # Converting image to a binary image 
    # ( black and white only image). 
    _, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY) 
    
    # Detecting contours in image. 
    _, contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    # Going through every contours found in the image. 
    for cnt in contours : 
    
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
    
        # draws boundary of contours. 
        cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)  
    
        # Used to flatted the array containing 
        # the co-ordinates of the vertices. 
        n = approx.ravel()  
        i = 0
    
        for j in n : 
            if(i % 2 == 0): 
                x = n[i] 
                y = n[i + 1] 
    
                # String containing the co-ordinates. 
                string = str(x) + " " + str(y)  
    
                if(i == 0): 
                    # text on topmost co-ordinate. 
                    cv2.putText(img2, "Arrow tip", (x, y), 
                                    font, 0.5, (255, 0, 0))  
                else: 
                    # text on remaining co-ordinates. 
                    cv2.putText(img2, string, (x, y),  
                            font, 0.5, (0, 255, 0))  
            i = i + 1
    # Showing the final image. 
    cv2.imshow('image2', img2)  
    
    # Exiting the window if 'q' is pressed on the keyboard. 
    if cv2.waitKey(0) & 0xFF == ord('q'):  
        cv2.destroyAllWindows() 
    #boxes, classes = predict_box_resnet(img)
    #draw = img.copy()
    #draw_detections(draw, boxes, classes)
    #print(boxes)
    #print(classes)
    #cv2.imshow('image', draw)
    #cv2.waitKey()

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