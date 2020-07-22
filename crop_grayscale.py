import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import io
import os
import sys
import urllib
import cv2
import time
import random
from PIL import Image
from collections import Counter


def save_annotations_and_classes(file_dir, objectsDF, classes=['card']):
  ANNOTATIONS_FILE = os.path.join(file_dir, "annotations.csv")
  CLASSES_FILE = os.path.join(file_dir, 'classes.csv')

  objectsDF.to_csv(ANNOTATIONS_FILE, index=False, header=None)
  classes = set(classes)

  with open(CLASSES_FILE, 'w') as f:
    for i, line in enumerate(sorted(classes)):
      f.write('{},{}\n'.format(line,i))


def get_xml(in_dir1, in_dir2):
  xml_files = []
  i = 0
  for file in os.listdir(in_dir1):
      if file.endswith(".xml"):
          if i < 600:
            print(i)
          else:
            xml_files.append(os.path.join(in_dir1, file))
          i += 1

  for file in os.listdir(in_dir2):
      if file.endswith(".xml"):
          xml_files.append(os.path.join(in_dir2, file))
  return xml_files


def create_cropped_annotations(in_dir1, in_dir2, out_dir):
  xml_files = get_xml(in_dir1, in_dir2)

  scale = 1
  delta = 0
  ddepth = cv2.CV_16S

  objects = []
  i = 0
  classes = []
  for sample in xml_files:
    tree = ET.parse(sample)
    root = tree.getroot()

    img_name = root.find('filename').text
    img_folder = root.find('folder').text

    if img_folder == 'FOLDER':
      img_folder = in_dir2
    else:
      img_folder = in_dir1
    img_name = os.path.join(img_folder, img_name)
    #image = cv2.imread(img_name)

    for elem in root.findall('object'):
      if i % 1000 == 0:
        print("sample {} of {}".format(i, len(xml_files)*6))
      obj = {}

      name = "{}.jpg".format(i)
      filename = os.path.join(out_dir, name)
      obj['image_name'] = filename
      top_x = int(elem.find('bndbox').find('xmin').text)
      top_y = int(elem.find('bndbox').find('ymin').text)
      bottom_x = int(elem.find('bndbox').find('xmax').text)
      bottom_y = int(elem.find('bndbox').find('ymax').text)
      
      #crop = image[top_y:bottom_y, top_x:bottom_x]
      #res_crop = cv2.resize(crop, (80, 80), interpolation = cv2.INTER_AREA)

      #gray = cv2.cvtColor(res_crop, cv2.COLOR_BGR2GRAY)

      #grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
      #grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
      
      #abs_grad_x = cv2.convertScaleAbs(grad_x)
      #abs_grad_y = cv2.convertScaleAbs(grad_y)
      
      #edged = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

      #edged = cv2.Canny(gray, 30, 200)

      obj['class_name'] = elem.find('name').text.lower()
      if obj['class_name'][0] in ['a', 'j', 'q', 'k'] and not ('IMG' in img_name):
        print("skip")
      else: 
        if obj['class_name'] not in classes:
          classes.append(obj['class_name'])
        objects.append(obj)
        #cv2.imwrite(filename, edged)
        i += 1
  

  objectsDF = pd.DataFrame(objects)
  print(objectsDF)
  save_annotations_and_classes(out_dir, objectsDF, classes=classes)
  return objectsDF, classes


def augment_dataset(dataset_path):
  annotations_path = os.path.join(dataset_path, "annotations.csv")
  objectsDF = pd.read_csv(annotations_path, names=['image_name', 'class_name'])
  labels_counter = Counter(objectsDF['class_name'].tolist())

  high_classes = []
  for idx, val in labels_counter.items():
    if idx[0] not in ['a', 'k', 'q', 'j']:
      high_classes.append(val)

  high_classes = np.array(high_classes)
  avg = int(np.average(high_classes, axis=0))
  print("average occurence of high classes: {}".format(avg))
  a_rows = []
  k_rows = []
  j_rows = []
  q_rows = []
  for index, row in objectsDF.iterrows():
    if row['class_name'][0] == 'a':
      a_rows.append(row)
    elif row['class_name'][0] == 'k':
      k_rows.append(row)
    elif row['class_name'][0] == 'j':
      j_rows.append(row)
    elif row['class_name'][0] == 'q':
      q_rows.append(row)

  def insert_row(idx, df, df_insert):
    return df.iloc[:idx, ].append(df_insert).append(df.iloc[idx:, ]).reset_index(drop = True)
  
  new_df = objectsDF.copy()
  for i in range(0, avg*5): # avg for 4 different aces
    new_idx = random.randint(0, objectsDF.shape[0])
    new_df = insert_row(new_idx, new_df, a_rows[i%len(a_rows)])
    new_idx = random.randint(0, objectsDF.shape[0])
    new_df = insert_row(new_idx, new_df, k_rows[i%len(k_rows)])
    new_idx = random.randint(0, objectsDF.shape[0])
    new_df = insert_row(new_idx, new_df, j_rows[i%len(j_rows)])
    new_idx = random.randint(0, objectsDF.shape[0])
    new_df = insert_row(new_idx, new_df, q_rows[i%len(q_rows)])
  print("old shape: {}".format(objectsDF.shape))
  print("new shape: {}".format(new_df.shape))
  ANNOTATIONS_FILE = os.path.join(dataset_path, "annotations_augmented.csv")
  new_df.to_csv(ANNOTATIONS_FILE, index=False, header=None)


read_dir1 = "card_dataset_resized"
read_dir2 = "playing-cards-dataset/test_zipped"
write_dir = "classifier_test"
objectsDF, classes = create_cropped_annotations(in_dir1=read_dir1, in_dir2=read_dir2, out_dir=write_dir)
#augment_dataset("./classifier_train")