import numpy as nps
from model.model import RedShiftClassificationModel
from keras.callbacks import TensorBoard
from time import time
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from astropy.visualization import make_lupton_rgb
import numpy as np


class Train():
    def __init__(self, batch_size=32, epochs=100):
        self.batch_size = batch_size
        self.epochs = epochs
        np.random.seed(32)

    def execute(self, model, dataset):
        X = dataset['X']
        y = dataset['y']
        y_cats = self.to_categorical(y)
        model.fit(X, y_cats, epochs=self.epochs, validation_split=0.15)
        # model.save_weights('epochs7500-25-weights.h5')

    def plot_hist(self, y):
        print(y.flatten().shape)
        sns.distplot(y.flatten(), kde=True)
        plt.show()

    def random_set_viz(self, X, y, num=25, display_channel=1):
        random_idxes = np.random.choice(np.random.permutation(X.shape[0]), size=num)
        print(random_idxes)
        random_samples = X[random_idxes]
        random_targets = y[random_idxes]
        random_cats = self.to_categorical(random_targets)
        num_rows_fig = num // 5
        fig, axes_list = plt.subplots(num_rows_fig, ncols=5)
        print(len(axes_list))
        k = 0
        print(random_samples.shape)
        for i in range(num_rows_fig):
            for j in range(5):
                # print(random_samples[k, :, :, 1])
                to_disp_img = make_lupton_rgb(random_samples[k, :, :, 0],
                                              random_samples[k, :, :, 1],
                                              random_samples[k, :, :, 2])
                axes_list[i][j].imshow(to_disp_img)
                axes_list[i][j].set_title("Z: {0}, Cat: {1}"
                                          .format(random_targets[k], random_cats[k]))
                k += 1
        plt.subplots_adjust(wspace=.4, hspace=.4)
        plt.show()

    def to_categorical(self, y, max_y=0.4, num_possible_classes=32):
        """Convert continuous redshift values into classes"""
        one_step = max_y / num_possible_classes
        y_cats = []
        # print(one_step)
        for values in y:
            y_cats.append(int(values[0]/one_step))
        return y_cats

    def data_generator(self, X, y, batch_size):
        X1, y1 = list(), list()
        n = 0
        while 1:
            for sample, label in zip(X, y):
                n += 1
                X1.append(sample)
                y1.append(label)
                if n == batch_size:
                    yield [[np.array(X1)], y1]
                    n = 0
                    X1, y1 = list(), list()


if __name__ == "__main__":
    from pickle import load
    with open('data/images/combined_dataset.pkl', 'rb') as pkl:
        dataset = load(pkl)
    model = RedShiftClassificationModel((64, 64, 5), 32)
    train_ins = Train(batch_size=32, epochs=25)
    train_ins.execute(model, dataset)
    # train_ins.plot_hist(dataset['y'])
