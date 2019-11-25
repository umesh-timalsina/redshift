# Editing "TrainValidate" Implementation
#
# The 'execute' method will be called when the operation is run
# Editing "Train" Implementation
#
# The 'execute' method will be called when the operation is run
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from matplotlib import pyplot as plt


class TrainValidate():
    def __init__(self, model, epochs=25, batch_size=32):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model
        np.random.seed(32)
        return

    def execute(self, dataset):
        model = self.model
        model.summary()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])
        X = dataset['X']
        y = dataset['y']
        y_cats = self.to_categorical(y)
        model.fit(X, y_cats,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  validation_split=0.15,
                  callbacks=[PlotLosses()])
        return model.get_weights()

    def to_categorical(self, y, max_y=0.4, num_possible_classes=32):
        one_step = max_y / num_possible_classes
        y_cats = []
        for values in y:
            y_cats.append(int(values[0] / one_step))
        return y_cats

    def datagen(self, X, y):
        # Generates a batch of data
        X1, y1 = list(), list()
        n = 0
        while 1:
            for sample, label in zip(X, y):
                n += 1
                X1.append(sample)
                y1.append(label)
                if n == self.batch_size:
                    yield [[np.array(X1)], y1]
                    n = 0
                    X1, y1 = list(), list()


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.i += 1

        self.update()

    def update(self):
        plt.clf()
        plt.title("Training Loss")
        plt.ylabel("CrossEntropy Loss")
        plt.xlabel("Epochs")
        plt.plot(self.x, self.losses, label="loss")
        plt.legend()
        plt.show();