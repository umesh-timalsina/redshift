"""
Let us assume that we are training around 50,000
images, inorder to apply the swarp tool it takes around one second
for each band. This means that 50000 * 5 = 2,50,000 secs (More than 3 days)
So, here we try to create a process pool of preparing dataset.

1. the updated meta for the images should be divided into
   number of processes. This will make sure that every process
   is working in a different file
2. Based on the number of cores, fire up a process pool.
3. Each process should execute the swarp wrappers.
4. Get the training set from each process and append them together
"""
import multiprocessing
from prepare_dataset import DatasetPreparer
import pandas as pd
import numpy as np
from copy import deepcopy


class ParallelDataSetPreparer():
    """Call the DatasetPreparer in parallel"""
    def __init__(self, images_meta, num_process=None):
        if num_process is None:
            self.num_proc = multiprocessing.cpu_count()
            if self.num_proc <= 1:
                self.num_proc = 1
        else:
            self.num_proc = num_process
        print('Initializing a pool with {} processes...'.format(self.num_proc))
        self.images_meta = images_meta
        self.process_pool = multiprocessing.Pool(
                                processes=self.num_proc)

    def execute(self, swarp_config_file, **kwargs):
        """Execute the dataset preparation pipeline in parallel
           args:
            swarp_config_file: the swarp config file
            **kwargs: The arguments to the dataset preparer's
                      prepare_dataset method
        """
        dataset_preparers = []
        image_metas = self._partition_meta()
        # Pass One: divid the data, prepare the dataset
        for i in range(self.num_proc):
            dataset_preparer = DatasetPreparer(config_file=swarp_config_file,
                                               images_meta=image_metas[i],
                                               meta_is_df=True)
            dataset_preparers.append(dataset_preparer)

            if not type(kwargs['filename']) == list:
                cached_filename = kwargs['filename']  # We need this again
                kwargs['filename'] = list()
            part_name = cached_filename.split('.')[0]\
                + '-PART{}.'.format(str(i))\
                + cached_filename.split('.')[1]
            kwargs['filename'].append(part_name)

            print('Executing in part, the partfile name is {}'.format(part_name))
        print(kwargs)
        results = []
        for i in range(self.num_proc):
            result = self.process_pool.apply_async(
                    dataset_preparers[i].prepare_dataset,
                    kwds={
                        'img_size': kwargs['img_size'],
                        'compressed': kwargs['compressed'],
                        'dump_pickle': kwargs['dump_pickle'],
                        'filename': kwargs['filename'][i]
                    })
            results.append(result)
        self.process_pool.close()
        self.process_pool.join()


        # Calling cleanup in one of them is ok
        dataset_preparers[0].cleanup()

    def _partition_meta(self):
        """Parition the images meta into smaller dfs"""
        whole_meta = pd.read_csv(self.images_meta)
        splitted_images_meta = np.array_split(whole_meta, self.num_proc)
        assert(len(splitted_images_meta) == self.num_proc)
        return splitted_images_meta
