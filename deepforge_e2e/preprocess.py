import os

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

from utils.swarp_string import swarp_config_string

class ExecuteSwarp():
    def __init__(self, img_size=64):
        """Initialize with Swarp Configuration"""
        self.swarp_file = '.swarp.conf'
        with open('.swarp.conf', 'w') as swarp_file:
            swarp_file.write(swarp_config_string)
        self.img_size = img_size
        print('Initialized with swarp file')
        self.images_meta = None

    def cleanup(self, compressed=True):
        """Remove the fits files produced during the operation"""
        data_dir = "/".join(self.images_meta.iloc[0]['u_loc'].split('/')[:-1])
        if compressed:
            os.system('rm -rf {}/*.fits'.format(data_dir))
        os.system('rm -rf *.fits')

    def execute(self, meta_df):
        self.images_meta = meta_df
        if self._check_imgs():
            X = []
            y = []
            for _, row in self.images_meta.iterrows():
                center = self.hmsdms_string(row['ra'], row['dec'])
                target_redshift = row['z']
                y.append(row['z'])
                fits_loc = row[-5:]
                one_example = self._return_datacube(fits_loc, self.img_size, center)
                X.append(one_example)
            X = np.array(X)
            assert (X.shape == (len(self.images_meta), self.img_size, self.img_size, 5))
            data_dir = '/'.join(fits_loc[0].split('/')[0:3])
            print(data_dir)
            self.cleanup()
            dataset = {
                'X': X,
                'y': y
            }
            return dataset

    def _return_datacube(self, fits_loc, img_size, center, compressed=True):
        """For a given set of fits files, return a
                  set (img_size, img_size, len(fits_loc)) matrix
        """
        data_mat = None
        for fits_file in fits_loc:
            if compressed:
                os.system('bzip2 -dk {}'.format(fits_file))
                file_loc = ".".join(fits_file.split('.')[:-1])
                print('Uncompressed FITS files is {}'.format(fits_loc))
            else:
                file_loc = fits_file
            ret = os.system('swarp {0}[0] -c {1} -CENTER {2},{3} \
                                       -IMAGEOUT_NAME {4} -WEIGHTOUT_NAME {5}'
                            .format(file_loc, self.swarp_file,
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
            print('Done Creating a {}*{}*{} datacube'.format(img_size, img_size, len(fits_loc)))
        assert (data_mat.shape == (img_size, img_size, len(fits_loc)))
        return data_mat

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