import cv2
import os

from .constants import SEGMENTED_FOLDER, EXTENSION


class ImageReader:
    def __init__(self, img_dir, manual_dir, mask_dir):
        self.img_dir = img_dir
        self.manual_dir = manual_dir
        self.mask_dir = mask_dir

        self.img_name = None
        self.img = None
        self.manual = None
        self.mask = None

    def read_img(self, img_name):
        self.img_name = img_name
        self.img = cv2.imread(os.path.join(self.img_dir, img_name))
        return self.img

    def read_manual(self, img_name):
        manual_path = os.path.join(self.manual_dir, img_name.split('.')[0] + '.tif')
        self.manual = cv2.imread(manual_path, cv2.IMREAD_GRAYSCALE)
        return self.manual

    def read_mask(self, img_name):
        fov_mask_path = os.path.join(self.mask_dir, img_name.split('.')[0] + '_mask.tif')
        self.mask = cv2.imread(fov_mask_path, cv2.IMREAD_GRAYSCALE)
        return self.mask

    def save_segmented(self, segmented_img, model_name):
        segmented_img_name = f"{self.img_name.split('.')[0]}_{model_name}.{EXTENSION}"
        segmented_path = os.path.join(SEGMENTED_FOLDER, segmented_img_name)
        if not os.path.exists(SEGMENTED_FOLDER):
            os.makedirs(SEGMENTED_FOLDER)
        cv2.imwrite(segmented_path, segmented_img)

    def get_img(self):
        return self.img

    def get_manual(self):
        return self.manual

    def get_mask(self):
        return self.mask
