import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    iter = np.arange(len(train_losses)) + 1
    iter = list(map(int, iter))
    plt.figure(0)
    plt.plot(iter, train_losses, '-b', label='Train loss')
    plt.plot(iter, valid_losses, '-m', label='Val Loss')
    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title('Loss Curves')
    plt.savefig("./output/plots/lossCurves.png")  # should before show method

    plt.figure(1)
    plt.plot(iter, train_accuracies, '-b', label='Train Accuracy')
    plt.plot(iter, valid_accuracies, '-m', label='Val Accuracy')
    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title('Accuracy Curves')
    plt.savefig("./output/plots/accCurves.png")  # should before show method



def plot_confusion_matrix(results, class_names):
    y_true = [n[0] for n in results]
    y_pred = [n[1] for n in results]
    cf = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(cf, display_labels=class_names)
    plt.rcParams["figure.figsize"] = (14, 10)
    disp.plot()
    plt.title('Normalized Confusion Matrix')
    plt.savefig("./output/plots/confMat.png")
