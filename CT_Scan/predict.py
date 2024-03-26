
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pydicom as dicom
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("test")

    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    test_x = glob("Dataset/test/*/*/*.dcm")
    print(f"Test: {len(test_x)}")

    for x in tqdm(test_x):
        dir_name = os.path.split(os.path.split(os.path.split(x)[0])[0])[1]
        name = dir_name + "_" + os.path.splitext(os.path.basename(x))[0]

        """ Read the image """
        image = dicom.dcmread(x).pixel_array
        image = np.expand_dims(image, axis=-1)
        image = image/np.max(image) * 255.0
        x = image/255.0
        x = np.concatenate([x, x, x], axis=-1)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        mask = model.predict(x)[0]
        mask = mask > 0.5
        mask = mask.astype(np.int32)
        mask = mask * 255

        cat_images = np.concatenate([image, mask], axis=1)
        cv2.imwrite(f"test/{name}.png", cat_images)