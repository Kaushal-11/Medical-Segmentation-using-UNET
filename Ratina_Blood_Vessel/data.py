import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion, CoarseDropout
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def load_data(path):
    """ X = Images and Y = masks """

    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images), desc="Augmenting Data"):
        """ Extracting names """
        name = os.path.splitext(os.path.basename(x))[0]

        x_img = cv2.imread(x, cv2.IMREAD_COLOR)
        y_mask = imageio.imread(y)
        
        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x_img, mask=y_mask)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x_img, mask=y_mask)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x_img, mask=y_mask)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = GridDistortion(p=1)
            augmented = aug(image=x_img, mask=y_mask)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x_img, mask=y_mask)
            x5 = augmented['image']
            y5 = augmented['mask']

            X = [x_img, x1, x2, x3, x4, x5]
            Y = [y_mask, y1, y2, y3, y4, y5]

        else:
            X = [x_img]
            Y = [y_mask]

        index = 0
        for img, mask in zip(X, Y):
            img = cv2.resize(img, (W, H))
            mask = cv2.resize(mask, (W, H))

            if len(X) == 1:
                img_name = f"{name}.jpg"
                mask_name = f"{name}.jpg"
            else:
                img_name = f"{name}_{index}.jpg"
                mask_name = f"{name}_{index}.jpg"

            img_path = os.path.join(save_path, "image", img_name)
            mask_path = os.path.join(save_path, "mask", mask_name)

            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "D:/Machine Learning/Medical_Segmentation/Ratina_Blood_Vessel/"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Creating directories """
    create_dir("D:/Machine Learning/Medical_Segmentation/Ratina_Blood_Vessel/new_data/train/image")
    create_dir("D:/Machine Learning/Medical_Segmentation/Ratina_Blood_Vessel/new_data/train/mask")
    create_dir("D:/Machine Learning/Medical_Segmentation/Ratina_Blood_Vessel/new_data/test/image")
    create_dir("D:/Machine Learning/Medical_Segmentation/Ratina_Blood_Vessel/new_data/test/mask")

    augment_data(train_x, train_y, "D:/Machine Learning/Medical_Segmentation/Ratina_Blood_Vessel/new_data/train/", augment=True)
    augment_data(test_x, test_y, "D:/Machine Learning/Medical_Segmentation/Ratina_Blood_Vessel/new_data/test/", augment=True)
