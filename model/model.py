# from keras.layers import Sequential
from keras import Model
from keras.layers import (Input, Conv2D, AveragePooling2D,
                          concatenate, Dense, PReLU, Flatten)
# from keras.layers.merge import concatenate
import numpy as np


class RedShiftClassificationModel(Model):
    def __init__(self,
                 input_img_shape,
                 num_redshift_classes):
        """Initialize the model"""
        # Input Layer Galactic Images
        image_input = Input(shape=input_img_shape)
        # Convolution Layer 1
        conv_1 = Conv2D(64,
                        kernel_size=(5, 5),
                        padding='same',
                        activation=PReLU())
        conv_1_out = conv_1(image_input)

        # Pooling Layer 1
        pooling_layer1 = AveragePooling2D(pool_size=(2, 2),
                                          strides=2,
                                          padding='same')
        pooling_layer1_out = pooling_layer1(conv_1_out)

        # Inception Layer 1
        inception_layer1_out = self.add_inception_layer(pooling_layer1_out,
                                                        num_f1=48,
                                                        num_f2=64)

        # Inception Layer 2
        inception_layer2_out = self.add_inception_layer(inception_layer1_out,
                                                        num_f1=64,
                                                        num_f2=92)

        # Pooling Layer 2
        pooling_layer2 = AveragePooling2D(pool_size=(2, 2),
                                          strides=2,
                                          padding='same')
        pooling_layer2_out = pooling_layer2(inception_layer2_out)

        # Inception Layer 3
        inception_layer3_out = self.add_inception_layer(pooling_layer2_out, 92, 128)

        # Inception Layer 4
        inception_layer4_out = self.add_inception_layer(inception_layer3_out, 92, 128)

        # Pooling Layer 3
        pooling_layer3 = AveragePooling2D(pool_size=(2, 2),
                                          strides=2,
                                          padding='same')
        pooling_layer3_out = pooling_layer3(inception_layer4_out)

        # Inception Layer 5
        inception_layer5_out = self.add_inception_layer(pooling_layer3_out,
                                                        92, 128,
                                                        kernel_5=False)

        # input_to_pooling = cur_inception_in
        input_to_dense = Flatten(
                            data_format='channels_last')(inception_layer5_out)
        print(input_to_dense.shape)
        model_output = Dense(units=num_redshift_classes, activation='softmax')(
                 Dense(units=num_redshift_classes, activation='relu')(
                       input_to_dense))

        super(RedShiftClassificationModel, self).__init__(
            inputs=[image_input], outputs=model_output)
        self.summary()

    def add_inception_layer(self,
                            input_weights,
                            num_f1,
                            num_f2,
                            kernel_5=True):
        """These convolutional layers take care of the inception layer"""
        # Conv Layer 1 and Layer 2: Feed them to convolution layers 5 and 6
        c1 = Conv2D(num_f1, kernel_size=(1, 1), padding='same', activation=PReLU())
        c1_out = c1(input_weights)
        if kernel_5:
            c2 = Conv2D(num_f1, kernel_size=(1, 1), padding='same', activation=PReLU())
            c2_out = c2(input_weights)

        # Conv Layer 3 : Feed to pooling layer 1
        c3 = Conv2D(num_f1, kernel_size=(1, 1), padding='same', activation=PReLU())
        c3_out = c3(input_weights)

        # Conv Layer 4: Feed directly to concat
        c4 = Conv2D(num_f2, kernel_size=(1, 1), padding='same', activation=PReLU())
        c4_out = c4(input_weights)

        # Conv Layer 5: Feed from c1, feed to concat
        c5 = Conv2D(num_f2, kernel_size=(3, 3), padding='same', activation=PReLU())
        c5_out = c5(c1_out)

        # Conv Layer 6: Feed from c2, feed to concat
        if kernel_5:
            c6 = Conv2D(num_f2, kernel_size=(5, 5), padding='same', activation=PReLU())
            c6_out = c6(c2_out)

        # Pooling Layer 1: Feed from conv3, feed to concat
        p1 = AveragePooling2D(pool_size=(2, 2), strides=1, padding='same')
        p1_out = p1(c3_out)

        if kernel_5:
            return concatenate([c4_out, c5_out, c6_out, p1_out])
        else:
            return concatenate([c4_out, c5_out, p1_out])


if __name__ == "__main__":
    from model_utils import save_model_image
    rscm = RedShiftClassificationModel((64, 64, 5), 1024)
    save_model_image(rscm, 'redshiftmodel.png')
    # print(rscm.predict(np.random.rand(1, 64, 64, 5)).shape)
    # rscm.prepare_for_training()
