from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

BASE_DIR = '/home/catfished/deepforge-dev/contrib/sdss-data'
MAX_REDSHIFT_VALUES = 0.4
MAX_DERED_PETRO_MAG = 17.8
REDSHIFT_KEY = 'z'
DEREDENED_PETRO_KEY = 'dered_petro_r'
EBV_KEY = 'EBV'


class DataSetSampler:
    """Sampler for the dataset

    Parameters
    ----------
    seed : int, default = 42
        random seed for numpy
    cube : np.ndarray
        the datacube
    labels : np.ndarray
        the labels array

    Attributes
    ----------
    cube : np.ndarray
        The numpy array of the base dataset
    labels : np.ndarray
        The base directory for the dataset
    tgt_indices : np.ndarray
        The intersection of the indexes in the dataset with redshifts in the desired deredened petro mags
    """
    def __init__(self,
                 seed=42,
                 cube=None,
                 labels=None):
        np.random.seed(seed)
        if cube is None:
            self.cube = np.load((Path(BASE_DIR) / 'cube.npy').resolve(), mmap_mode='r')
            self.labels = np.load((Path(BASE_DIR) / 'labels.npy').resolve(), mmap_mode='r')
        else:
            self.cube = cube
            self.labels = labels
        self.tgt_indices = self._find_intersection()

    def _find_intersection(self):
        """filter the redshift values"""
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
        num_samples = self.tgt_indices.shape[0] * percentage // 100
        print(f'sampling at {percentage} % results in {num_samples} for '
              f'training and {self.tgt_indices.shape[0]-num_samples} testing.')

        sample_idxes_train = np.random.choice(self.tgt_indices, num_samples, replace=False)
        sample_idxes_test = np.setdiff1d(self.tgt_indices, sample_idxes_train)

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

    def plot_histogram(self,
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
