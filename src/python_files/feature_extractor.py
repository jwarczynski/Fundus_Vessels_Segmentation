import numpy as np
import pandas as pd

from skimage.filters import roberts, sobel, scharr
from scipy import ndimage as nd
import cv2


class FeatureExtractor:
    def __init__(self, preprocessed_img):
        self.preprocessed_img = preprocessed_img
        self.features = pd.DataFrame()

    def extract_features(self):
        self.add_edge_features()
        self.add_gaussian_filter_features()
        self.add_median_features()
        self.add_gabor_features()

        return self.features

    def add_edge_features(self):
        self.add_original_image()
        self.add_canny_edges()
        self.add_roberts_edges()
        self.add_sobel_edges()
        self.add_scharr_edges()

    def add_original_image(self):
        self.features['Original Image'] = self.preprocessed_img.flatten()

    def add_canny_edges(self):
        canny_edges = cv2.Canny(self.preprocessed_img, 100, 200)
        self.features['Canny Edge'] = canny_edges.flatten()

    def add_roberts_edges(self):
        roberts_edges = roberts(self.preprocessed_img)
        self.features['Roberts'] = roberts_edges.flatten()

    def add_sobel_edges(self):
        sobel_edges = sobel(self.preprocessed_img)
        self.features['Sobel'] = sobel_edges.flatten()

    def add_scharr_edges(self):
        scharr_edges = scharr(self.preprocessed_img)
        self.features['Scharr'] = scharr_edges.flatten()

    def add_gaussian_filter_features(self):
        gaussian_s3 = nd.gaussian_filter(self.preprocessed_img, sigma=3)
        gaussian_s7 = nd.gaussian_filter(self.preprocessed_img, sigma=3)
        self.features['Gaussian s3'] = gaussian_s3.flatten()
        self.features['Gaussian s7'] = gaussian_s7.flatten()

    def add_median_features(self):
        median_s3 = nd.median_filter(self.preprocessed_img, size=3)
        median_s7 = nd.median_filter(self.preprocessed_img, size=7)
        self.features['Median s3'] = median_s3.flatten()
        self.features['Median s7'] = median_s7.flatten()

    def add_gabor_features(self):
        num = 1
        kernels = self.generate_gabor_kernels(kernel_size=9)
        for kernel in kernels:
            filtered_img = self.apply_gabor_filter(kernel)
            filtered_img = filtered_img.flatten()
            gabor_label = 'Gabor' + str(num)
            self.add_gabor_feature(gabor_label, filtered_img)
            num += 1

    def generate_gabor_kernels(self, kernel_size):
        kernel_params = [
            (theta / 4. * np.pi, sigma, lamda, gamma)
            for theta in range(2)
            for sigma in (1, 3)
            for lamda in np.arange(0, np.pi, np.pi / 4)
            for gamma in (0.05, 0.5)
        ]
        kernels = [
            cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
            for theta, sigma, lamda, gamma in kernel_params
        ]
        return kernels

    def apply_gabor_filter(self, kernel):
        return cv2.filter2D(self.preprocessed_img, cv2.CV_8UC3, kernel)

    def add_gabor_feature(self, gabor_label, filtered_img):
        self.features[gabor_label] = filtered_img

