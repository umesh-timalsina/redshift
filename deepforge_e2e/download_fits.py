import os
import time
from tempfile import mkdtemp

import numpy as np

SDSS_IMAGE_URL = """https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/{rerun}\
/{run}/{camcol}/frame-{band}-{run_str}-{camcol}-{field}.fits.bz2"""


class DownloadFITS():
    """Download the fits files from SDSS Server"""
    def __init__(self, url=None, num_images=10, seed=32, only_scan=True):
        if not url:
            self.url = SDSS_IMAGE_URL
        self.num_images = num_images
        np.random.seed(seed)
        self.meta = None
        self.data_loc = mkdtemp()
        self.only_scan = only_scan

    def execute(self, meta_df):
        if self.meta is None:
            self.meta = self._randomize_meta(meta_df)
        dl_count = self._download_fits(only_scan=self.only_scan)
        print('downloaded {} Images'.format(dl_count))
        return self.meta

    def _randomize_meta(self, meta_df):
        """Ramdomly resample the dataset"""
        sample_frac = self.num_images / meta_df.shape[0]
        df = meta_df.sample(frac=sample_frac)
        return df

    def _download_fits(self, only_scan=True):
        """Given the galaxies download images"""
        images_path = {
            'u_loc': [],
            'g_loc': [],
            'r_loc': [],
            'i_loc': [],
            'z_loc': []
        }
        loc_itr = ['u_loc', 'g_loc', 'r_loc', 'i_loc', 'z_loc']
        dl_count = 0
        for i, galaxy in self.meta.iterrows():
            to_dl = self._get_formatted_urls(galaxy['rerun'],
                                             galaxy['run'],
                                             galaxy['camcol'],
                                             galaxy['field'])
            i = 0
            for band_loc, url in zip(loc_itr, to_dl):
                # self.download_image(band_loc, url)
                file_name = os.path.join(self.data_loc, url.split('/')[-1])
                if only_scan:
                    # self.meta[band_loc].append(file_name)
                    images_path[band_loc].append(file_name)
                    dl_count += 1
                else:
                    dl_count += self.download_image(band_loc, url)
                    images_path[band_loc].append(file_name)
        print('Number of Images Downloaded: ', dl_count)
        # self.meta = pd.concat([self.meta, urls_df], axis=1)
        for loc in loc_itr:
            assert (len(images_path[loc]) == self.meta.shape[0])
            self.meta[loc] = images_path[loc]
        self.meta.to_csv(os.path.join(self.data_loc, 'updated_meta_{}.csv'.format(time.time())))
        print('Successfully saved the csv file with updated meta in {}'.format(self.data_loc))
        return dl_count

    def download_image(self, band_loc, url):
        """Download Images from the sas server
                    Important: This is very slow, write
                    a shell script for this
        """
        file_name = os.path.join(self.data_loc, url.split('/')[-1])
        ret = os.system('wget -O {0} {1}'.format(file_name, url))
        if ret == 0:
            print('Successfully Downloaded -> ', file_name)
            return 1
        else:
            print('Failed to download -> ', file_name)
            return 0

    def _get_formatted_urls(self, rerun, run, camcol, field):
        """For this galaxy return all possible image urls"""
        possible_bands = ['u', 'g', 'r', 'i', 'z']
        run_arr = list(str(run).strip())

        if len(run_arr) < 6:
            run_arr = ['0']*(6-len(run_arr)) + run_arr
        run_str = ''.join(run_arr)

        field_arr = list(str(field).strip())

        if len(field_arr) < 4:
            field_arr = ['0'] * (4 - len(field_arr)) + field_arr

        field_str = ''.join(field_arr)

        ret_list = list()
        for band in possible_bands:
            ret_list.append(self.url.format(run=str(run),
                                            run_str=run_str,
                                            rerun=str(rerun),
                                            field=field_str,
                                            camcol=str(camcol),
                                            band=band))
        return ret_list