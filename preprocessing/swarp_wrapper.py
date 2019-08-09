"""
A wrapper around the swarp tool. With options to change few confugurations
Since the swarp tool already uses a parallel pipeline execution scheme,
no effort should be made to parallelize here :)
"""

import os
import configparser
from astropy.io import fits
import numpy as np
import bz2


class SwarpExecutor():
    def __init__(self, config_file=None):
        """Initialize the class.
            params:
                config_file: path to a config file for swarp
        """
        if config_file is not None:
            print('Using the provided config file ->', config_file)
            self.config_file = config_file
        else:
            ret_val = os.system('swarp -dd > .swarp.conf')
            if ret_val != 0:
                raise Exception('Error generating the swap configuration file')
            self.config_file = '.swarp.conf'

    def return_datacube(self, fits_loc, img_size, compressed=True):
        """For a given set of fits files, return a
           set (img_size, img_size, len(fits_loc)) matrix
        """
        data_mat = None
        for fits_file in fits_loc:
            if compressed:
                os.system('bzip2 -dk {}'.format(fits_file))
                file_loc = ".".join(fits_file.split('.')[:-1])
                print('Uncompressed Fits file is -> ', file_loc)
            else:
                file_loc = fits_file
            ret = os.system('swarp {0} -c {1}'.format(file_loc, self.config_file))
            if ret != 0:
                raise Exception('Error processing the input image')
            print('Done resampling =>', fits_file)
            with fits.open('./coadd.fits') as _fits_data:
                one_channel = np.expand_dims(_fits_data[0].data, axis=-1)
                if data_mat is None:
                    data_mat = one_channel
                else:
                    data_mat = np.concatenate((data_mat, one_channel), axis=-1)
        assert(data_mat.shape == (img_size, img_size, len(fits_loc)))
        return data_mat


if __name__ == "__main__":
    import pandas as pd
    meta = pd.read_csv('../data/images/updated_meta.csv')
    se = SwarpExecutor(config_file='./.swarp.conf')
    for i in range(5):
        cur_data = meta.iloc[i]
        cur_data = list(cur_data[-5:])
        my_mat = se.return_datacube(cur_data, 64)
        print(my_mat.shape)
        print(np.equal(np.zeros((64, 64, 5)), my_mat))
