import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import re

cropped_path = "./to_augment"
annotations = os.path.join(cropped_path, 'annotations_augmented.csv')

def visualize_dataset(testOrTrain, annotations_file):
    testDF = pd.read_csv(annotations_file, names=['image_name', 'class_name'])
    labels = testDF['class_name'].tolist()
    print("found {} labels".format(len(labels)))

    #labels_count = Counter(labels)
    #labels_list = [val for _, val in labels_count.items()]
    #print(labels_list)
    fig, ax = plt.subplots(tight_layout=True, figsize=(18, 9))
    plt.title("Class Distribution "+testOrTrain)
    ax.hist(labels, bins=np.arange(0, 53))
    ax.set_ylabel('Occurence')
    ax.set_xlabel('Data')

    s = r'(.*)annotations.*'
    path = re.findall(s, annotations_file)[0]
    fig_name = os.path.join(path, "ClassDistribution_"+testOrTrain+".png")
    fig.savefig(fig_name, format='png', dpi=600)
    plt.show()

visualize_dataset("train_augmented", annotations)