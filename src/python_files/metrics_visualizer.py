import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


class MetricsVisualizer:
    def __init__(self, ground_truth, predicted):
        self.ground_truth = ground_truth
        self.predicted = predicted

        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None

        self.accuracy = None
        self.sensitivity = None
        self.specificity = None

    def calculate_metrics(self):
        self.TP = np.sum(np.logical_and(self.ground_truth == 255, self.predicted == 255))
        self.TN = np.sum(np.logical_and(self.ground_truth == 0, self.predicted == 0))
        self.FP = np.sum(np.logical_and(self.ground_truth == 0, self.predicted != 0))
        self.FN = np.sum(np.logical_and(self.ground_truth == 255, self.predicted != 255))

        self.accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        self.sensitivity = self.TP / (self.TP + self.FN)
        self.specificity = self.TN / (self.TN + self.FP)

    def visualize_metrics(self):
        self.tabulate_metrics()
        self.show_expert_predicted_and_misclassified()
        self.show_confusion_matrix_plots()

    def tabulate_metrics(self):
        headers = ["Accuracy", "Sensitivity", "Specificity", "TP", "TN", "FP", "FN"]
        data = [[self.accuracy, self.sensitivity, self.specificity, self.TP, self.TN, self.FP, self.FN]]
        print(tabulate(data, headers=headers))

    def show_expert_predicted_and_misclassified(self):
        misclassification_mask = self.calculate_misclassification_mask()

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(self.ground_truth, cmap='gray')
        plt.title('Expert Mask')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(self.predicted, cmap='gray')
        plt.title('Segmentation Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(misclassification_mask, cmap='gray')
        plt.title('Misclassification Mask')
        plt.axis('off')

        plt.show()

    def calculate_misclassification_mask(self):
        misclassified_mask = np.zeros((*self.ground_truth.shape, 3), dtype=np.uint8)
        misclassified_mask[np.logical_and(self.predicted > 128, self.ground_truth <= 128)] = [255, 0, 0]
        misclassified_mask[np.logical_and(self.predicted > 128, self.ground_truth > 128)] = [255, 255, 255]
        misclassified_mask[np.logical_and(self.predicted < 128, self.ground_truth > 128)] = [0, 0, 255]
        return misclassified_mask

    def visualize_misclassification_mask(self):
        misclassified_mask = self.calculate_misclassification_mask()
        plt.imshow(misclassified_mask)

    def show_confusion_matrix_plots(self):
        labels = ["TP", "TN", "FP", "FN"]
        values = [self.TP, self.TN, self.FP, self.FN]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].bar(labels, values)
        axs[0].set_title('Confusion Matrix - Bar Plot')

        cm = confusion_matrix(self.ground_truth.flatten(), self.predicted.flatten())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axs[1])
        axs[1].set_title('Confusion Matrix - Heatmap')

        plt.tight_layout()
        plt.show()

    def bar_plot_confusion_matrix(self):
        labels = ["TP", "TN", "FP", "FN"]
        values = [self.TP, self.TN, self.FP, self.FN]
        plt.bar(labels, values)
        plt.show()

    def show_confusion_matrix(self):
        cm = confusion_matrix(self.ground_truth.flatten(), self.predicted.flatten())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
