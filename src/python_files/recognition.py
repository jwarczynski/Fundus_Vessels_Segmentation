import cv2
import numpy as np
from skimage import filters, morphology


def threshold_image(image, threshold):
    thresholded_image = np.where(image < threshold, 0, 255)
    return thresholded_image


def preprocess(image):
    _, green_chanel, _ = cv2.split(image)
    img = cv2.medianBlur(green_chanel, 7)

    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(7, 7))
    img = clahe.apply(img)

    return img


def process_image(preprocessed_img):
    img = filters.frangi(preprocessed_img, gamma=0.8, beta=0.15)
    img = (img * 255).astype(np.uint8)
    return img


def post_process_image(processed_img):
    img = cv2.medianBlur(processed_img, 3)
    img = threshold_image(img, 35)
    img = np.array(img, bool)
    img = morphology.remove_small_objects(img, min_size=128) * 255
    img = img.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = img.astype(np.uint8)

    return img


def apply_fov_mask(img, fov_mask):
    img = cv2.bitwise_and(img, img, mask=fov_mask.astype(np.uint8))
    return img


def segment_vessels(image, fov_mask):
    preprocessed_img = preprocess(image)
    processed_img = process_image(preprocessed_img)
    postprocessed_img = post_process_image(processed_img)
    segmentation_mask = apply_fov_mask(postprocessed_img, fov_mask)
    return segmentation_mask


def calculate_misclassified_mask(segmentation_mask, expert_mask):
    misclassified_mask = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)
    misclassified_mask[np.logical_and(segmentation_mask > 128, expert_mask <= 128)] = [255, 0, 0]
    misclassified_mask[np.logical_and(segmentation_mask > 128, expert_mask > 128)] = [255, 255, 255]
    misclassified_mask[np.logical_and(segmentation_mask < 128, expert_mask > 128)] = [0, 0, 255]
    # misclassified_mask[np.logical_and(segmentation_mask > 128, expert_mask > 128)] = [0, 255, 0]
    return misclassified_mask


def calculate_metrics(expert_mask, generated_mask):
    TP = np.sum(np.logical_and(expert_mask == 255, generated_mask == 255))
    TN = np.sum(np.logical_and(expert_mask == 0, generated_mask == 0))
    FP = np.sum(np.logical_and(expert_mask == 0, generated_mask == 255))
    FN = np.sum(np.logical_and(expert_mask == 255, generated_mask == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return TP, TN, FP, FN, accuracy, sensitivity, specificity
