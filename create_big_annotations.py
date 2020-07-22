import os
import pandas as pd
import xml.etree.ElementTree as ET

def create_annotations(file_dir):
  xml_files = []
  for file in os.listdir(file_dir):
      if file.endswith(".xml"):
          xml_files.append(os.path.join(file_dir, file))

  objects = []
  i = 0
  classes = []
  for sample in xml_files:
    if i % 100 == 0:
      print("sample {} of {}".format(i, len(xml_files)))
    tree = ET.parse(sample)
    root = tree.getroot()

    img_name = root.find('filename').text
    for elem in root.findall('object'):
      obj = {}
      obj['image_name'] = os.path.join(file_dir ,img_name)
      obj['top_x'] = elem.find('bndbox').find('xmin').text
      obj['top_y'] = elem.find('bndbox').find('ymin').text
      obj['bottom_x'] = elem.find('bndbox').find('xmax').text
      obj['bottom_y'] = elem.find('bndbox').find('ymax').text
      obj['class_name'] = elem.find('name').text.lower()
      objects.append(obj)
      if obj['class_name'] not in classes:
        classes.append(obj['class_name'])
    i += 1

  objectsDF = pd.DataFrame(objects)
  print(objectsDF)
  return objectsDF, classes


def save_annotations_and_classes(file_dir, objectsDF, filename="annotations.csv", classes=['card']):
  ANNOTATIONS_FILE = os.path.join(file_dir, filename)
  CLASSES_FILE = os.path.join(file_dir, 'classes.csv')

  objectsDF.to_csv(ANNOTATIONS_FILE, index=False, header=None)
  classes = set(classes)

  with open(CLASSES_FILE, 'w') as f:
    for i, line in enumerate(sorted(classes)):
      f.write('{},{}\n'.format(line,i))


train_dir = "./card_dataset_resized/"
objectsDF, classes = create_annotations(train_dir)
save_annotations_and_classes(train_dir, objectsDF, classes=classes)