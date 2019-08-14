from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from model.model import RedShiftClassificationModel
from keras.callbacks import TensorBoard
from time import time
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from astropy.visualization import make_lupton_rgb


class Train():
    def __init__(self, batch_size=32, epochs=100):
        self.batch_size = batch_size
        self.epochs = epochs
        np.random.seed(32)

    def execute(self, model, dataset):
        X = dataset['X']
        y = dataset['y']
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.1, random_state=32)
        # self.plot_hist(np.asarray(train_y))
        train_y = np.array(train_y)
        # self.random_set_viz(train_X, train_y)
        
        train_y_cats = self.to_categorical(train_y)
        valid_y_cats = self.to_categorical(valid_y)
        # tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
        # self.datagen.flow(x=X, y=y_cats, shuffle=True)
        steps_per_epoch_train = train_X.shape[0]//self.batch_size
        steps_per_epoch_valid = valid_X.shape[0]//self.batch_size

        # model.fit_generator(self.data_generator(train_X, train_y_cats, self.batch_size),
        #                     steps_per_epoch=steps_per_epoch_train,
        #                     epochs=self.epochs, verbose=1,
        #                     validation_data=self.data_generator(
        #                         valid_X, valid_y_cats, self.batch_size),
        #                     validation_steps=steps_per_epoch_valid)
        model.fit(train_X, train_y_cats, epochs=25)

        # model.save('10000-100Epochs.h5')

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
        num_rows_fig = num // 3
        fig, axes_list = plt.subplots(num_rows_fig, ncols=3)
        print(len(axes_list))
        k = 0
        print(random_samples.shape)
        for i in range(num_rows_fig):
            for j in range(3):
                # print(random_samples[k, :, :, 1])
                to_disp_img = make_lupton_rgb(random_samples[k, :, :, 2],
                                              random_samples[k, :, :, 3],
                                              random_samples[k, :, :, 4])
                axes_list[i][j].imshow(to_disp_img)
                axes_list[i][j].set_title("Z: {0}, Cat: {1}"
                                          .format(random_targets[k], random_cats[k]))
                k += 1
        plt.subplots_adjust(wspace=.4, hspace=.4)
        plt.show()

    def to_categorical(self, y, max_y=0.4, num_possible_classes=1024):
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
    model = RedShiftClassificationModel((64, 64, 5), 1024)
    train_ins = Train(batch_size=32)
    train_ins.execute(model, dataset)
