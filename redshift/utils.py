import os
import warnings
import math
from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

BASE_DIR = os.environ['BASE_DIR']
MAX_REDSHIFT_VALUES = 0.4
MAX_DERED_PETRO_MAG = 17.8
REDSHIFT_KEY = 'z'
DEREDENED_PETRO_KEY = 'dered_petro_r'
EBV_KEY = 'EBV'


def has_astropy():
    try:
        import astropy
        del astropy
        return True
    except ImportError:
        return False


class DataSetSampler:
    """Sampler for the dataset

    This class is used to load the Dataset(~53GB) or a mmap,
    sample it and return the indexes according to the values.

    Parameters
    ----------
    seed : int, default = None
        random seed for numpy
    cube : np.ndarray, default=None
        the datacube
    labels : np.ndarray, default=None
        the labels array

    Attributes
    ----------
    cube : np.ndarray
        The numpy array of the base dataset
    labels : np.ndarray
        The base directory for the dataset
    tgt_indices : np.ndarray
        The intersection of the indexes in the dataset with
        redshifts in the desired deredened petro mags

    Notes
    -----
    If no mmaps are provided, the dataset is assumed to be in BASE_DIR.
    """
    def __init__(self,
                 seed=42,
                 cube=None,
                 labels=None):
        if cube is None:
            if BASE_DIR is None:
                raise ValueError(
                    'Please provide the mmaps, or set environment for BASE_DIR'
                )
            self.cube = np.load((Path(BASE_DIR) / 'cube.npy').resolve(), mmap_mode='r')
            self.labels = np.load((Path(BASE_DIR) / 'labels.npy').resolve(), mmap_mode='r')
        else:
            self.cube = cube
            self.labels = labels
        self.tgt_indices = self._find_intersection()
        self.seed = seed

    def _find_intersection(self):
        """find the galaxies in the dataset with desired values"""
        redshifts = self.labels[REDSHIFT_KEY]
        dered_petro_mag = self.labels[DEREDENED_PETRO_KEY]
        (idxes_redshifts,) = (redshifts <= MAX_REDSHIFT_VALUES).nonzero()
        (idxes_dered,) = (dered_petro_mag <= MAX_DERED_PETRO_MAG).nonzero()
        intersection = np.intersect1d(idxes_redshifts,
                                      idxes_dered,
                                      return_indices=False)
        print(f'There are {intersection.shape[0]} galaxies with redshift '
              f'values between (0, {MAX_REDSHIFT_VALUES}] and '
              f'dered_petro_mag between (0, {MAX_DERED_PETRO_MAG}].')

        return intersection

    def return_k_fold_indices(self,
                              percentage=2,
                              num_folds=10,
                              return_test_indices=False):
        """Return k-folds of intersecting indices from the dataset

        Parameters
        ----------
        percentage : int, default=2
            The percentage of the dataset to return indices from
        num_folds : int, default=10
            The num of folds to return
        return_test_indices : bool, default=False
            Whether or not to return test indices for the dataset

        Returns
        -------
        np.ndarray
            Array of indexes in the dataset with folds
        """
        num_samples = int(self.tgt_indices.shape[0] * percentage) // 100
        print(f'sampling at {percentage} % results in {num_samples} for '
              f'training and {self.tgt_indices.shape[0] - num_samples} testing.')

        sample_idxes_train, sample_idxes_test = train_test_split(
            self.tgt_indices,
            random_state=self.seed,
            train_size=num_samples
        )
        k_folds = np.array_split(sample_idxes_train, num_folds)
        folds_dict = {}
        for j in range(num_folds):
            to_concat = [k_folds[i] for i in range(len(k_folds)) if i != j]
            folds_dict[f'train_fold_{j+1}'] = np.concatenate(to_concat)
            folds_dict[f'valid_fold_{j+1}'] = k_folds[j]

        if not return_test_indices:
            return folds_dict

        return folds_dict, sample_idxes_test

    def return_samples(self,
                       percentage=2,
                       train=True):
        """Return the samples from the dataset

        Parameters
        ----------
        percentage: int, default=2
            The percentage of the dataset to return for sampling
        train: bool, default=True
            If True, return the test set
        """
        num_samples = int(self.tgt_indices.shape[0] * percentage // 100)
        print(f'sampling at {percentage} % results in {num_samples} for '
              f'training and {self.tgt_indices.shape[0]-num_samples} testing.')

        sample_idxes_train, sample_idxes_test = train_test_split(
            self.tgt_indices,
            random_state=self.seed,
            train_size=num_samples
        )

        assert (sample_idxes_test.shape[0] + sample_idxes_train.shape[0] == self.tgt_indices.shape[0])
        assert np.intersect1d(sample_idxes_train, sample_idxes_test).size == 0

        if train:
            datacube = self.cube[sample_idxes_train]
            z_truth = self.labels[REDSHIFT_KEY][sample_idxes_train]
            ebv = self.labels[EBV_KEY][sample_idxes_train]
        else:
            datacube = self.cube[sample_idxes_test]
            z_truth = self.labels[REDSHIFT_KEY][sample_idxes_test]
            ebv = self.labels[EBV_KEY][sample_idxes_train]

        return {
            'x': datacube,
            'ebv': ebv,
            'Y': z_truth
        }

    def save_histogram(self,
                       bins=180,
                       kde=True,
                       hist=True,
                       filename='dataset.png'):
        """Plot the histogram of the dataset"""
        redshifts = self.labels['z'][self.tgt_indices]
        sns.distplot(redshifts.flatten(),
                     bins=bins,
                     kde=kde,
                     hist=hist)
        plt.xlabel('Redshift values')
        plt.ylabel('Frequency')
        plt.savefig(filename)


class MMapSequence(Sequence):
    """Sequence to load mmap in mini batches, with capabilities to mutate

    Parameters
    ----------
    labels : np.mmap,
        The mmap array for labels on the dataset
    cube : np.mmap
        The mmap array for cube on the dataset
    idxes : np.ndarray
        The indexes of interest for this sequence
    batch_size : int, default=128
        The batch size
    flip_prob : float, default=0.2
        The probability of flipping (augmentation)
    rotate_prob : float, default=0.2
        The probability of 90 degree rotation (augmentation)
    max_value : float, default=0.4
        The maximum redshift value in the sequences
    num_bins : int default=180
        The number of bins to divide redshift values to
    """
    def __init__(self,
                 labels,
                 cube,
                 idxes,
                 batch_size=128,
                 flip_prob=0.2,
                 rotate_prob=0.2,
                 max_value=0.4,
                 num_bins=180):
        self.labels = labels
        self.cube = cube
        self.idxes = idxes
        self.batch_size = batch_size
        self.num_batches = math.ceil(idxes.shape[0] / self.batch_size)
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_value = max_value
        self.num_bins = num_bins
        print(f'batch size: {self.batch_size}, '
              f'number of batches : {self.num_batches}, '
              f'dataset size: {self.idxes.shape}')
        self.batches = self._generate_batch_indexes()
        self.flip_indexes = None
        self.rotate_indexes = None
        self.on_epoch_end()

    def on_epoch_end(self):
        self.flip_indexes, self.rotate_indexes = self.augument_indexes()

    def _generate_batch_indexes(self):
        batches = {}

        for i in range(self.num_batches):
            batches[i] = self.idxes[i*self.batch_size: i*self.batch_size+self.batch_size]
        return batches

    def _get_batch_at(self, index, is_categorical=True):
        indexes = self.batches.get(index)
        datacube = self.cube[indexes]
        z_truth = self.labels[REDSHIFT_KEY][indexes]
        if is_categorical:
            z_truth = to_categorical(
                z_truth,
                max_value=self.max_value,
                num_bins=self.num_bins,
            )
        ebv = np.expand_dims(self.labels[EBV_KEY][indexes], axis=-1)
        return (datacube, ebv), z_truth

    def augument_indexes(self):
        flips, rotate = set(), set()
        for index in self.idxes:
            if np.random.random() < self.flip_prob:
                flips.add(index)
            if np.random.random() < self.rotate_prob:
                rotate.add(index)
        return flips, rotate

    def plot_batch(self, index=0, filename=None):
        # FIXME : Currently hardcoded for a batchsize of 128
        if self.batch_size != 128:
            warnings.warn('This method is hard coded for batchsize of 128 for now')
        if not has_astropy():
            warnings.warn('Plotting galaxies requires the astropy package')
            return
        else:
            from astropy.visualization import make_lupton_rgb
        (images, ebvs), z_truths = self._get_batch_at(index,
                                                      is_categorical=True)
        cols = 10
        rows = math.ceil(self.batch_size / cols)
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(rows, cols))
        fig.suptitle(f'batch {index} from the dataset')
        for row_count in range(rows):
            for col_count in range(cols):
                axes[row_count][col_count].tick_params(
                    axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False
                )
                axes[row_count][col_count].tick_params(
                    axis='y',
                    which='both',
                    left=False,
                    right=False,
                    labelleft=False
                )

        row_count, col_count = 0, 0
        plt.subplots_adjust(hspace=0.8, wspace=0.2)

        for i, (image, ebv, z_truth) in enumerate(zip(images, ebvs, z_truths)):
            im = make_lupton_rgb(image[:, :, 3],
                                 image[:, :, 2],
                                 image[:, :, 1],
                                 Q=10, stretch=0.5)
            axes[row_count, col_count].imshow(im)
            axes[row_count, col_count].set_xlabel(f'z = {str(round(z_truth, 6))} \n'
                                                  f'ebv = {ebv}',
                                                  fontsize=6)

            col_count += 1
            if col_count == cols:
                row_count += 1
                col_count = 0

        for j in range(col_count, cols):
            axes[row_count][j].axis('off')

        if filename is None:
            filename = f'batch_{index}.png'
        plt.savefig(filename, bbox_inches='tight')

    def __getitem__(self, index):
        (actual_cube, ebv), z_truth = self._get_batch_at(index)
        indexes = self.batches.get(index)
        datacube = []
        for i, index in enumerate(indexes):
            one_cube = actual_cube[i]
            if index in self.rotate_indexes:
                one_cube = np.rot90(one_cube)
            if index in self.flip_indexes:
                one_cube = np.flip(one_cube)
            datacube.append(one_cube)

        return (np.array(datacube), ebv), z_truth

    def __len__(self):
        return self.num_batches


def to_categorical(targets,
                   max_value=0.4,
                   num_bins=180,
                   verify_cats=False):
    """Returns a categorical mapping of a numpy array

    Parameters
    ----------
    targets : np.ndarray
        A np.ndarray of targets
    max_value : float, default=0.4
        The maximum possible value in the targets
    num_bins : int, defualt=180
        The number of classes to divide into
    verify_cats : bool, default=False
        If true perform a verbose assertion on categorical mapping(useful for testing)

    Returns
    -------
    np.ndarray
        The categorical mapping of targets

    Notes
    -----
    The categories range from [9, num_bins)
    """
    bins = np.linspace(0,
                       max_value,
                       num_bins + 1,
                       endpoint=True)
    y_cats = (np.digitize(targets, bins))
    if verify_cats:
        for n in range(targets.size):
            assert bins[y_cats[n]-1] <= targets[n] < bins[y_cats[n]]
            print(f'{bins[y_cats[n]-1]} <= Value: {targets[n]} -> '
                  f'Class: {y_cats[n] - 1} < {bins[y_cats[n]]}')
    if len(y_cats.shape) == 1:
        y_cats = np.expand_dims(y_cats, axis=-1)
    return (y_cats - 1).astype(int)
