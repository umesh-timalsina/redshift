from astropy.visualization import make_lupton_rgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pickle import load
import numpy as np
from collections import OrderedDict


class VisualizePredictions():
    """Visualize Some of the predictions of the model
        params:
            num(int): the number of plots
            num_classes(int): the number of output classes
    """
    def __init__(self, num=5, num_classes=32):
        self.num = num
        self.num_classes = num_classes
        np.random.seed(32)

    def execute(self, model, dataset):
        """Execute the plotting
            params:
                model: the model
                dataset: the dataset
        """
        X = dataset['X']
        gnd = dataset['y']
        _, X, _, gnd = train_test_split(X, gnd, test_size=0.2)
        print(self.num_classes)
        start = 0.4 / (2*self.num_classes)
        bin_dist = 0.4 / self.num_classes
        bins = []
        for i in range(self.num_classes):
            bins.append(start)
            start += bin_dist
        print(bins)
        idxes = np.random.randint(X.shape[0], size=self.num)+10
        X = X[idxes]
        gnd = gnd[idxes]
        y = model.predict(X)
        print(y.shape)
        fig, axes = plt.subplots(nrows=self.num, ncols=1)
        for image, pred, gnd_val, ax in zip(X, y, gnd, axes):
            ax.plot(pred, label='Softmax output')
            z = np.sum(pred.reshape(1, self.num_classes) * bins)
            print(z, gnd_val)
            ax.text(0.9, 0.8,
                    "z(predicted): {0:.6},\n z(real): {1:.6}".format(z, gnd_val[0]),
                    style='italic',
                    ha='center',
                    va='center',
                    transform=ax.transAxes)
            ax.vlines(z/0.0125, ymin='.2', ymax='.8', label='prediction', linestyles='dashed', color='r')
            ax.vlines(gnd_val/0.0125, ymin='.2', ymax='.8', label='ground values', linestyles='dashed', color='g')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.figlegend(by_label.values(), by_label.keys())
        plt.show()