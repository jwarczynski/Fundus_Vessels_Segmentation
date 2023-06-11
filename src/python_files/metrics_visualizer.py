import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


class MetricsVisualizer:
    def __init__(self):
        self.TP = []
        self.TN = []
        self.FP = []
        self.FN = []

        self.accuracy = []
        self.sensitivity = []
        self.specificity = []

        self.mean_TP = None
        self.mean_TN = None
        self.mean_FP = None
        self.mean_FN = None

        self.mean_accuracy = None
        self.mean_sensitivity = None
        self.mean_specificity = None

        self.predictions = []
        self.ground_truths = []
        self.file_names = []

    def calculate_metrics(self, ground_truth, predicted, file_name):
        self.TP.append(np.sum(np.logical_and(ground_truth == 255, predicted == 255)))
        self.TN.append(np.sum(np.logical_and(ground_truth == 0, predicted == 0)))
        self.FP.append(np.sum(np.logical_and(ground_truth == 0, predicted != 0)))
        self.FN.append(np.sum(np.logical_and(ground_truth == 255, predicted != 255)))

        self.accuracy.append((self.TP[-1] + self.TN[-1]) / (self.TP[-1] + self.TN[-1] + self.FP[-1] + self.FN[-1]))
        self.sensitivity.append(self.TP[-1] / (self.TP[-1] + self.FN[-1]))
        self.specificity.append(self.TN[-1] / (self.TN[-1] + self.FP[-1]))

        self.predictions.append(predicted)
        self.ground_truths.append(ground_truth)
        self.file_names.append(file_name)

    def calculate_mean_metrics(self):
        self.mean_TP = np.mean(self.TP)
        self.mean_TN = np.mean(self.TN)
        self.mean_FP = np.mean(self.FP)
        self.mean_FN = np.mean(self.FN)

        self.mean_accuracy = np.mean(self.accuracy)
        self.mean_sensitivity = np.mean(self.sensitivity)
        self.mean_specificity = np.mean(self.specificity)

    def visualize(self):
        self.tabulate_metrics()
        for i in range(len(self.file_names)):
            self.visualize_single(i)

    def visualize_single(self, i):
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        axs[0, 0].imshow(self.predictions[i], cmap='gray')
        axs[0, 0].set_title(f"{self.file_names[i]} Prediction")

        axs[0, 1].imshow(self.ground_truths[i], cmap='gray')
        axs[0, 1].set_title(f"{self.file_names[i]} Ground Truth")

        axs[0, 2].imshow(self.calculate_misclassification_mask(i), cmap='gray')
        axs[0, 2].set_title(f"{self.file_names[i]} Misclassification Mask")

        cm = confusion_matrix(self.ground_truths[i].flatten(), self.predictions[i].flatten())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axs[1, 0])
        axs[1, 0].set_title(f"{self.file_names[i]} Confusion Matrix")

        axs[1, -1].remove()

        axs[1, 1].bar(["TP", "TN", "FP", "FN"], [self.TP[i], self.TN[i], self.FP[i], self.FN[i]])
        axs[1, 1].set_title(f"{self.file_names[i]} Confusion Matrix - Bar Plot")

        plt.tight_layout()
        plt.show()

    def tabulate_metrics(self):
        headers = ["Image", "Accuracy", "Sensitivity", "Specificity", "TP", "TN", "FP", "FN"]
        self.calculate_mean_metrics()
        # data = [self.accuracy, self.sensitivity, self.specificity, self.TP, self.TN, self.FP, self.FN]
        data = [[file_name, accuracy, sensitivity, specificity, TP, TN, FP, FN]
                for file_name, accuracy, sensitivity, specificity, TP, TN, FP, FN
                in zip(self.file_names, self.accuracy, self.sensitivity, self.specificity, self.TP, self.TN, self.FP, self.FN)]
        data.append(
            ["mean", self.mean_accuracy, self.mean_sensitivity, self.mean_specificity,
             self.mean_TP, self.mean_TN, self.mean_FP, self.mean_FN
             ]
        )
        print(tabulate(data, headers=headers))

    def calculate_misclassification_mask(self, i=0):
        misclassified_mask = np.zeros((*self.ground_truths[i].shape, 3), dtype=np.uint8)
        misclassified_mask[np.logical_and(self.predictions[i] > 128, self.ground_truths[i] <= 128)] = [255, 0, 0]
        misclassified_mask[np.logical_and(self.predictions[i] > 128, self.ground_truths[i] > 128)] = [255, 255, 255]
        misclassified_mask[np.logical_and(self.predictions[i] < 128, self.ground_truths[i] > 128)] = [0, 0, 255]
        return misclassified_mask

    def visualize_misclassification_mask(self, i=0):
        misclassified_mask = self.calculate_misclassification_mask(i)
        plt.imshow(misclassified_mask)

    def show_confusion_matrix_plots(self, i):
        labels = ["TP", "TN", "FP", "FN"]
        values = [self.TP[i], self.TN[i], self.FP[i], self.FN[i]]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].bar(labels, values)
        axs[0].set_title('Confusion Matrix - Bar Plot')

        cm = confusion_matrix(self.ground_truths[i].flatten(), self.predictions[i].flatten())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axs[1])
        axs[1].set_title('Confusion Matrix - Heatmap')

        plt.tight_layout()
        plt.show()

    def bar_plot_confusion_matrix(self, i=0):
        labels = ["TP", "TN", "FP", "FN"]
        values = [self.TP[i], self.TN[i], self.FP[i], self.FN[i]]
        plt.bar(labels, values)
        plt.show()

    def show_confusion_matrix(self, i=0):
        cm = confusion_matrix(self.ground_truths[i].flatten(), self.predictions[i].flatten())
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
