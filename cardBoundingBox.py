import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D, Activation, Add, BatchNormalization, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
import xml.etree.ElementTree as ET
import pandas as pd
import keras
import random

batch_size = 4
epochs = 10
output_layers=15
REDUCE_FACTOR = 4
IMG_HEIGHT = int(800/REDUCE_FACTOR)
IMG_WIDTH = int(600/REDUCE_FACTOR)
xml_data = glob.glob('playing-cards-dataset/train_zipped_resized/*.xml')
img_data = glob.glob('playing-cards-dataset/train_zipped_resized/*.jpg') #+ glob.glob('playing-cards-dataset/train_zipped_resized/*.JPG')


card_path = os.path.join('playing-cards-dataset', 'train_zipped_resized')
activation_function = tf.nn.leaky_relu

def create_resnet_model(num_of_layers=50):
    inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    block_2, block_3 = [4, 6]
    if num_of_layers is 101:
        block_2, block_3 = [4, 23]
    elif num_of_layers is 152:
        block_2, block_3 = [8, 36]


    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation_function)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, filters=[64, 64, 256], s=1)
    for i in range(3):
        x = id_block(x, [64, 64, 256])

    x = conv_block(x, filters=[128, 128, 512])
    for i in range(block_2):
        x = id_block(x, [128, 128, 512])
    
    x = conv_block(x, filters=[256, 256, 1024])
    for i in range(block_3):
        x = id_block(x, [256, 256, 1024])

    x = conv_block(x, filters=[512, 512, 2048])
    for i in range(3):
        x = id_block(x, [512, 512, 2048])

    x = AveragePooling2D((2,2))(x)
    x = Flatten()(x)
    #x = Dense(52, activation=tf.nn.relu)(x)

    output_activation = tf.nn.leaky_relu
    outputs = []
    for i in range(output_layers):
        outputs.append(Dense(4, activation=output_activation, name='out' + str(i))(x))

    
    res_net_model = tf.keras.Model(inputs, outputs=outputs, name='test_net')
    return res_net_model


def conv_block(input_data, filters, s=2):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1), strides=(s,s))(input_data)
    x = BatchNormalization(axis = 3)(x)
    x = Activation(activation_function)(x) 

    x = Conv2D(f2, (3, 3), strides=(1,1), padding = 'same')(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation(activation_function)(x) 

    x = Conv2D(f3, (1, 1), strides=(1,1), padding = 'valid')(x)
    x = BatchNormalization(axis = 3)(x)

    skip = Conv2D(f3, (1, 1), strides=(s,s), padding = 'valid')(input_data)
    skip = BatchNormalization(axis = 3)(skip)

    x = Add()([x, skip])
    x = Activation(activation_function)(x) 

    return x


def id_block(input_data, filters):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1), strides=(1,1), padding='valid')(input_data)
    x = BatchNormalization(axis = 3)(x)
    x = Activation(activation_function)(x) 

    x = Conv2D(f2, (3, 3), strides=(1,1), padding = 'same')(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation(activation_function)(x)

    x = Conv2D(f3, (1, 1), strides=(1,1), padding = 'valid')(x)
    x = BatchNormalization(axis = 3)(x)

    x = Add()([x, input_data])
    x = Activation(activation_function)(x) 

    return x

'''def save_annotations_and_classes(file_dir, objectsDF):
  ANNOTATIONS_FILE = os.path.join(file_dir, 'annotations_bounding_box.csv')
  CLASSES_FILE = os.path.join(file_dir, 'classes_bounding_box.csv')

  objectsDF.to_csv(ANNOTATIONS_FILE, index=False, header=None)
  classes = set(['xmin', 'ymin', 'xmax', 'ymax'])

  with open(CLASSES_FILE, 'w') as f:
    for i, line in enumerate(sorted(classes)):
      f.write('{},{}\n'.format(line,i))

def create_annotations(file_dir):
    xml_files = []
    for file in os.listdir(file_dir):
        if file.endswith(".xml"):
            xml_files.append(os.path.join(file_dir, file))

    objects = []
    i = 0
    for sample in xml_files:
        if i % 100 == 0:
          print("sample {} of {}".format(i, len(xml_files)))
        tree = ET.parse(sample)
        root = tree.getroot()

        img_name = root.find('filename').text
        #boxes = []
        for box in root.findall('.//bndbox'):
            obj = {}
            obj['image_name'] = str(os.path.join(file_dir ,img_name))
            obj['xmin'] = box.find('xmin').text
            obj['ymin'] = box.find('ymin').text
            obj['xmax'] = box.find('xmax').text
            obj['ymax'] = box.find('ymax').text
            obj['class_name'] = 'card'
            objects.append(obj)
        i += 1
    objectsDF = pd.DataFrame(objects)
    print(objectsDF)
    return objectsDF


def read_annotation_files(model):
    TRAIN_ANNOTATIONS_FILE = os.path.join('playing-cards-dataset', 'test_zipped', 'annotations_bounding_box.csv')

    y_cols = ['xmin', 'ymin', 'xmax', 'ymax']

    tmp_names = ['image_name'] + y_cols + ['class_name']

    trainDF = pd.read_csv(TRAIN_ANNOTATIONS_FILE, names=tmp_names)
    #grouped = trainDF.groupby('image_name', axis=1)
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, 
        #horizontal_flip=True, 
        #vertical_flip=True,
        #rotation_range=360,
        #shear_range=10,
        #zoom_range=0.2
    )

    #y_cols = ['xmin', 'ymin', 'xmax', 'ymax']
    train_generator = datagen.flow_from_dataframe(
        dataframe=trainDF,
        x_col='image_name',
        y_col=y_cols,
        class_mode='other',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        color_mode='rgb'
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    model_path = 'BoundingBox_Locator.h5'

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

    history = model.fit(train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=epochs,
                        #validation_data=test_generator
    )
    model.save(model_path)
'''


def model_train(model):
    model.compile(optimizer='adam',
                  loss=tf.losses.MAE,
                  metrics=['accuracy'])

    # train on all training data without smashing the RAM
    total_x = []
    total_y = []
    cnt = 0
    #print('Loading data: ')
    random.seed(42)
    random.shuffle(img_data)

    for img_filename in img_data:

        xml_name = img_filename.split('\\')[-1].split('.')[0] + ".xml"
        xml_filename = os.path.join(card_path, xml_name)

        tree = ET.parse(xml_filename)
        root = tree.getroot()
        single_y = []

        flip = bool(random.getrandbits(1))
        for elem in root.findall('object'):

            if flip:
                single_y.append([int(elem.find('bndbox').find('xmin').text)/REDUCE_FACTOR,
                                 (600-int(elem.find('bndbox').find('ymin').text))/REDUCE_FACTOR,
                                 int(elem.find('bndbox').find('xmax').text)/REDUCE_FACTOR,
                                 (600-int(elem.find('bndbox').find('ymax').text))/REDUCE_FACTOR])
            else:
                single_y.append([int(elem.find('bndbox').find('xmin').text)/REDUCE_FACTOR,
                                 int(elem.find('bndbox').find('ymin').text)/REDUCE_FACTOR,
                                 int(elem.find('bndbox').find('xmax').text)/REDUCE_FACTOR,
                                 int(elem.find('bndbox').find('ymax').text)/REDUCE_FACTOR])
        
        #random.shuffle(single_y)
        for i in range(output_layers-len(single_y)):
            single_y.append([0, 0, 0, 0])
        
        total_y.append(single_y)
        
        #img = cv2.imread(img_filename)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.Canny(img, 30, 200)
        #image = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)
        image = cv2.imread(img_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = np.array(tf.image.resize(plt.imread(img_filename), [IMG_HEIGHT, IMG_WIDTH]))
        if flip:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, (IMG_HEIGHT,IMG_WIDTH))

        total_x.append(image)
        #xml_data.remove(xml_filename)
        #img_data.remove(img_filename)
        if len(total_x) == len(img_data) and len(total_y) == len(img_data):
            print('------  NEXT DATA -------- '+str(cnt))

            if cnt > 0:
                model = tf.keras.models.load_model('BoundingBox_Locator.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})

            cnt += 1
            #break
            #out.update(progress(len(total_x), len(xml_data)))
            #print(total_y)

            x_input = np.array(total_x)

            y_output = {}
            for i in range(output_layers):
                y_output['out'+str(i)] = np.array([item[i] for item in total_y]) 

            y_input = np.array(total_y)
            total_x.clear()
            total_y.clear()
            model.fit(x_input,
                      y_output,
                      batch_size=batch_size, 
                      epochs=epochs, 
                      #steps_per_epoch=500
            )
            model.save('BoundingBox_Locator.h5')

def show_image_objects(img, predictions):
    image = cv2.imread(img)#read_image_bgr(img_path)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    #draw = cv2.resize(draw, (800, 800), interpolation=cv2.INTER_AREA)
    #draw = draw[0:600, 0:800]
    #draw = cv2.resize(draw, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
  
    boxes = []
    sub_box =[]
    for box in predictions:
        #print(boxes)
        sub_box.append(box[0][0]*REDUCE_FACTOR)
        sub_box.append(box[0][1]*REDUCE_FACTOR)
        sub_box.append(box[0][2]*REDUCE_FACTOR)
        sub_box.append(box[0][3]*REDUCE_FACTOR)
        boxes.append(sub_box)
        sub_box = []
  
    for box in boxes:
        if not np.isnan(np.array(box)).any():
            cv2.rectangle(draw, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,255,0), thickness=2)
    #draw_box(draw, box, color=(255, 255, 0))
  
    plt.axis('off')
    plt.imshow(draw)
    plt.show()



def model_predict():
    model = tf.keras.models.load_model('BoundingBox_Locator.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    files = glob.glob('playing-cards-dataset/train_zipped_resized/*.jpg')
    for i in range(0, 10):
        img = cv2.imread(files[i])
        #img = img[0:600, 0:800]
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.Canny(img, 30, 200)
        #test_image = cv2.resize(img, (800, 800), interpolation = cv2.INTER_AREA)
        #test_image = test_image[0:600, 0:800]
        test_image = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
        #test_image = np.array(tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]))
        predictions = model.predict(np.array([test_image]))
        print(predictions)
        show_image_objects(files[i], predictions)


train_dir = "playing-cards-dataset/train_zipped_resized"

#objectsDF = create_annotations(train_dir)
#save_annotations_and_classes(train_dir, objectsDF)

model = create_resnet_model(num_of_layers=50)
model_train(model)
model_predict()