# Set these environment variables for finding dataset location
import os
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint

from logger_factory import LoggerFactory
from utils import DataSetSampler, MMapSequence
from model.model import RedShiftClassificationModel

PERCENTAGE_TRAINING = 99
NUM_FOLDS = 5
CHECKPOINTS_DIR = './results-glorot-uniform-seed-72'
EPOCHS = 30

os.environ['BASE_DIR'] = os.environ.get('BASE_DIR', './dataset')


def main():
    dataset_sampler = DataSetSampler()
    k_folds_indices = dataset_sampler.return_k_fold_indices(
        percentage=PERCENTAGE_TRAINING,
        num_folds=NUM_FOLDS
    )
    logger = LoggerFactory.get_logger(__file__)
    logger.info('Performing cross validation with {}-folds'.format(NUM_FOLDS))

    model = RedShiftClassificationModel(
        (64, 64, 5),
        redenning_shape=(1,)
    )
    init_weights = model.get_weights()

    model.compile()
    for i in range(NUM_FOLDS):
        logger.info(f'Fold {i+1}')
        model.set_weights(init_weights)
        this_train_fold = f'train_fold_{i+1}'
        this_valid_fold = f'train_fold_{i+1}'
        triaining_sequence = MMapSequence(
            labels=dataset_sampler.labels,
            cube=dataset_sampler.cube,
            idxes=k_folds_indices[this_train_fold],
            batch_size=256
        )

        validation_sequence = MMapSequence(
            labels=dataset_sampler.labels,
            cube=dataset_sampler.cube,
            idxes=k_folds_indices[this_valid_fold],
            batch_size=256
        )

        if not (a := Path(f'{CHECKPOINTS_DIR}/{PERCENTAGE_TRAINING}-percent-of-dataset/'
                          f'{this_train_fold}/')).resolve().exists():
            os.makedirs(a, exist_ok=True)

        model.fit(
            triaining_sequence,
            validation_data=validation_sequence,
            epochs=EPOCHS,
            callbacks=[
                ModelCheckpoint(
                    filepath=f'{CHECKPOINTS_DIR}/{PERCENTAGE_TRAINING}-percent-of-dataset/'
                             f'{this_train_fold}/' + 'weights.{epoch:02d}.hdf5',
                    save_weights_only=True,
                    monitor='val_loss',
                    save_freq='epoch',
                    mode='min',
                    verbose=1,
                    save_best_only=True
                )
            ]
        )


if __name__ == '__main__':
    main()

