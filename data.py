import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob  # use to extract paths
from tqdm import tqdm  
import imageio  # use to read gif files
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion, CoarseDropout

def create_dir(path):   #create a directory
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):    #load data set
    """ X = Images and Y = masks """

    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):   #if augment == True, apply data augmentation on both images and masks and save them
    H = 512                                                 #if augment is false, just resize them and save them in the same folders
    W = 512

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting names """
        name = x.split("\\")[-1].split(".")[0] 

        #print(x," ",name)

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]


        
        if augment == True:     #if augment == True, apply data augmentation on both images and masks and save them
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            X = [x, x1, x2, x3, x4, x5]
            Y = [y, y1, y2, y3, y4, y5]

        else:       #if augment is false, just resize them and save them in the same folders
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):      #i = image, m = mask
            i = cv2.resize(i, (W, H))       #resize image
            m = cv2.resize(m, (W, H))       #resize mask

            if len(X) == 1:     # if there is one image
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name = f"{name}.jpg"
            else:               # if more than one image
                tmp_image_name = f"{name}_{index}.jpg"
                tmp_mask_name = f"{name}_{index}.jpg"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            #print(image_path," ",mask_path)

#new_data/train/image\DRIVE\training\images\31_training.jpg   new_data/train/mask\DRIVE\training\images\31_training.jpg
#new_data/train/image\DRIVE\training\images\32_training.jpg   new_data/train/mask\DRIVE\training\images\32_training.jpg
            
            ###################################
            #Convert BGR to RGB (OpenCV loads images in BGR format)
            #image_rgb = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)

            #Display the image using matplotlib
            #plt.imshow(image_rgb)
            #plt.axis('off')  # Turn off axis numbers and ticks
            #plt.show()
            ###################################

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path =  "D:\sem 5\image processing\DRIVE"      # "C:/Users/Asus/OneDrive/Desktop/imp/DRIVE"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")    #how many images
    print(f"Test: {len(test_x)} - {len(test_y)}")       #how many images   

    """ Creating directories """
    create_dir("new_data/train/image")
    create_dir("new_data/train/mask")
    create_dir("new_data/test/image")
    create_dir("new_data/test/mask")

    augment_data(train_x, train_y, "new_data/train/", augment=False)
    augment_data(test_x, test_y, "new_data/test/", augment=False)