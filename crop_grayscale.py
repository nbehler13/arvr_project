import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import io
import os
import urllib
import cv2
import time
from PIL import Image



card_path = "C:/Users/nickl/Pictures/ARVR_Projekt"


def save_annotations_and_classes(file_dir, objectsDF, classes=['card']):
  ANNOTATIONS_FILE = os.path.join(file_dir, "annotations.csv")
  CLASSES_FILE = os.path.join(file_dir, 'classes.csv')

  objectsDF.to_csv(ANNOTATIONS_FILE, index=False, header=None)
  classes = set(classes)

  with open(CLASSES_FILE, 'w') as f:
    for i, line in enumerate(sorted(classes)):
      f.write('{},{}\n'.format(line,i))


def get_xml(in_dir1):
  xml_files = []
  for file in os.listdir(in_dir1):
      if file.endswith(".xml"):
          xml_files.append(os.path.join(in_dir1, file))

  return xml_files


def create_cropped_annotations(in_dir1, out_dir):
  xml_files = get_xml(in_dir1)

  objects = []
  i = 0
  classes = []
  for sample in xml_files:
    tree = ET.parse(sample)
    root = tree.getroot()

    img_name = root.find('filename').text
    #img_folder = root.find('folder').text

    img_folder = in_dir1

    img_name = os.path.join(img_folder, img_name)
    image = cv2.imread(img_name)

    for elem in root.findall('object'):
      if i % 1000 == 0:
        print("sample {} of {}".format(i, len(xml_files)*4))
      obj = {}

      name = "{}.jpg".format(i)
      filename = os.path.join(out_dir, name)
      obj['image_name'] = filename
      top_x = int(elem.find('bndbox').find('xmin').text)
      top_y = int(elem.find('bndbox').find('ymin').text)
      bottom_x = int(elem.find('bndbox').find('xmax').text)
      bottom_y = int(elem.find('bndbox').find('ymax').text)
      
      crop = image[top_y:bottom_y, top_x:bottom_x]
      res_crop = cv2.resize(crop, (80, 80), interpolation = cv2.INTER_AREA)

      gray = cv2.cvtColor(res_crop, cv2.COLOR_BGR2GRAY)
      edged = cv2.Canny(gray, 30, 200)


      obj['class_name'] = elem.find('name').text.lower()
      if obj['class_name'][0] in ['k', 'a', 'j'] and 'IMG' not in img_name:
          continue

      objects.append(obj)
      if obj['class_name'] not in classes:
        classes.append(obj['class_name'])
      i += 1

      cv2.imwrite(filename, edged)

  objectsDF = pd.DataFrame(objects)
  print(objectsDF)
  save_annotations_and_classes(out_dir, objectsDF, classes=classes)
  return objectsDF, classes

read_dir1 = card_path
#read_dir2 = "train_zipped"
write_dir = "C:/Users/nickl/Pictures/card_dataset_cropped/"
objectsDF, classes = create_cropped_annotations(in_dir1=read_dir1, out_dir=write_dir)