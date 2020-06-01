import os
import unittest
import numpy as np

from .utils import to_categorical, DataSetSampler, MMapSequence


def get_lables_and_cubes():
    os.system('wget -O data.npz https://vanderbilt.box.com/shared/static/yio0378rwywv463657z8edxhamkrru7f.npz')
    dataset = np.load('data.npz', mmap_mode='r')
    labels, cube = dataset['labels'], dataset['cube']
    return labels, cube


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.np_array = np.array([0.0, 0.09, 0.06, 0.39])
        labels, cube = None, None
        if os.environ.get('BASE_DIR', None) is None:
            labels, cube = get_lables_and_cubes()
        cls.sampler = DataSetSampler(labels=labels,
                                     cube=cube)

    def test_to_categorical(self):
        y_cats = to_categorical(self.np_array,
                                max_value=0.4,
                                num_bins=10,
                                verify_cats=True)
        assert y_cats.shape == (4, 1), y_cats.shape
        assert np.allclose(y_cats, np.array([[0], [2], [1], [9]])), y_cats

    def test_dataset_preparer_categories(self):
        dataset = self.sampler.return_samples()
        targets = np.random.choice(dataset['Y'], size=100, replace=False)
        classes = to_categorical(targets,
                                 max_value=0.4,
                                 num_bins=180,
                                 verify_cats=True)
        assert classes.shape[0] == targets.shape[0]

    def test_sequencer(self):
        indexes, test_set = self.sampler.return_k_fold_indices(percentage=30,
                                                               num_folds=5,
                                                               return_test_indices=True)
        sequences = MMapSequence(
            cube=self.sampler.cube,
            labels=self.sampler.labels,
            idxes=indexes['train_fold_1']
        )
        (images, ebv), z = sequences._get_batch_at(1)
        assert images.shape == (sequences.batch_size, 64, 64, 5), images.shape
        assert np.intersect1d(sequences.idxes, test_set).size == 0
        for batch in range(sequences.num_batches-1):
            assert np.intersect1d(sequences.batches[batch],
                                  sequences.batches[batch+1]).size == 0

    @classmethod
    def tearDownClass(cls):
        os.system('rm -f data.npz')


if __name__ == '__main__':
    unittest.main()
