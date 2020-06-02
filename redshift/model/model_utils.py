from tensorflow.keras.utils import plot_model


def save_model_image(model, filename):
    """Save the keras model graph to a file"""
    plot_model(model,
               to_file=filename,
               show_shapes=True,
               show_layer_names=True,
               dpi=96)
