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
import glob
import pandas as pd
from pickle import dump
from astropy.coordinates import SkyCoord
from astropy import units as u


class DatasetPreparer():
    def __init__(self, config_file=None, images_meta=None, meta_is_df=False):
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
        if not meta_is_df:
            self.images_meta = pd.read_csv(images_meta, sep=',')
        else:
            self.images_meta = images_meta

    def cleanup(self, compressed=True):
        """Remove the fits files produced during the operation"""
        data_dir = "/".join(self.images_meta.iloc[0]['u_loc'].split('/')[:-1])
        if compressed:
            os.system('rm -rf {}/*.fits'.format(data_dir))
        os.system('rm -rf *.fits')

    def _check_imgs(self):
        """Check if all the images in the meta file are present"""
        for i in range(len(self.images_meta)):
            cur_files = self.images_meta.iloc[i]
            cur_files = list(cur_files[-5:])
            # print(cur_files)
            missing = False
            for _file in cur_files:
                # print(_file)
                if not os.path.exists(_file):
                    missing = True
            if missing:
                print('Some Files are missing')
                break
        if not missing:
            print('All the files are found')
            return True
        else:
            return False

    def prepare_dataset(self,
                        img_size=64,
                        compressed=True,
                        dump_pickle=True,
                        filename='dataset.pkl'):
        print('The data will be dumped to {}'.format(filename))
        if self._check_imgs():
            X = []
            y = []
            for _, row in self.images_meta.iterrows():
                # print(row['ra'], row['dec'], row[-5:])
                center = self.hmsdms_string(row['ra'], row['dec'])
                print(center)
                target_redshift = row['z']
                y.append([row['z']])
                fits_loc = row[-5:]
                one_example = self.return_datacube(fits_loc, 64, center)
                X.append(one_example)
            X = np.array(X)
        assert(X.shape == (len(self.images_meta), img_size, img_size, 5))
        data_dir = '/'.join(fits_loc[0].split('/')[0:3])
        print(data_dir)

        # Remove the fits files produced during the operation
        dataset = {
            'X': X,
            'y': y
            }
        print('Trying to pickle....')
        if dump_pickle:
            with open('{}/{}'.format(data_dir, filename), 'wb+') as data_fp:
                dump(dataset, data_fp)
            print('Successfully dumped dataset to {}'.format(data_dir+'/{}'.format(filename)))    
        return dataset

    def return_datacube(self, fits_loc, img_size, center, compressed=True):
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
            ret = os.system('swarp {0}[0] -c {1} -CENTER {2},{3} \
                            -IMAGEOUT_NAME {4} -WEIGHTOUT_NAME {5}'
                            .format(file_loc, self.config_file,
                                    center[0], center[1],
                                    'coadd-process{}.fits'.format(os.getpid()),
                                    'coadd.weight-process{}.fits'.format(os.getpid())))
            if ret != 0:
                raise Exception('Error processing the input image')
            print('Done resampling =>', fits_file)
            with fits.open('coadd-process{}.fits'.format(os.getpid())) as _fits_data:
                one_channel = np.expand_dims(_fits_data[0].data, axis=-1)
                if data_mat is None:
                    data_mat = one_channel
                else:
                    data_mat = np.concatenate((data_mat, one_channel), axis=-1)
            # os.system('rm -rf *.fits')
            print('Done Creating a {}*{}*{} datacube'.format(img_size, img_size, len(fits_loc)))
        assert(data_mat.shape == (img_size, img_size, len(fits_loc)))
        return data_mat

    def hmsdms_string(self, ra, dec):
        center_coord = SkyCoord(ra*u.degree, dec*u.degree)
        coords = center_coord.to_string('hmsdms').split(' ')
        for i in range(len(coords)):
            coords[i] = coords[i].replace('h', ':')
            coords[i] = coords[i].replace('m', ':')
            coords[i] = coords[i].replace('s', '')
            coords[i] = coords[i].replace('d', ':')
        print(coords)
        return tuple(coords)


if __name__ == "__main__":
    import pandas as pd
    # meta = pd.read_csv('../data/images/updated_meta.csv')
    se = DatasetPreparer(config_file='./.swarp.conf', images_meta=glob.glob('../data/images/updated_meta_*.csv')[0])
    se.prepare_dataset(64)
    se.cleanup()
