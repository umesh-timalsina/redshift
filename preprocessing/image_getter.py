#! /usr/bin/env python
"""Set of Utility Methods written for getting images
    before we feed it to the swarp tool.
    1. Note, there are close to 550000 galaxy objects returned from the images,
       So, for our case, we want to specify what percentage of the images
       to download (of course) randomly
    2. After we download the image we want to be able to put it
       in a common directory where it is processed
    3. By default this coud will download FITS For 100,000 galaxies.
    4. That's it
"""
import numpy as np
from scrapy import Selector
import requests
import os
import pandas as pd


class ImagesDownloader():

    def __init__(self, url, loc, num_images, csv_loc, seed=32):
        """A class with all the methods to download fits
            params:
                url: The url format string of the fits object
                for dr12 this is as shown in the example below:
                    https://dr12.sdss.org/sas/dr12/boss/photoObj/\
                    frames/301/[run]/[camcol]/frame-[ugriz]-[run]-[camcol]-[fieldno].fitz.bz2
                loc: a local directory to save the extracted fits file
                num_images: Number of images to download
                csv_loc: location of the csv meta file
                seed: the random seed to randomly choose 10000 entries from the meta file
        """
        self.url = url
        if not (os.path.exists(loc) and os.path.isdir(loc)):
            os.makedirs(loc)
        self.data_loc = loc
        np.random.seed(seed)  # Change this for a different set of files
        self._randomize_meta(num_images, csv_loc)

    def _randomize_meta(self, num_images, csv_loc):
        """Initialize the meta with randomly selected samples"""
        df = pd.read_csv(csv_loc)
        sample_frac = num_images/df.shape[0]
        df = df.sample(frac=sample_frac)
        # print(df.shape)
        self.meta = df

    def download_fits(self):
        """Given the CSV for galaxies, download and extract the fits files
           and save them to a particular location
        """
        self.meta['u_loc'] = np.NaN
        self.meta['g_loc'] = np.NaN
        self.meta['r_loc'] = np.NaN
        self.meta['i_loc'] = np.NaN
        self.meta['z_loc'] = np.NaN
        loc_itr = ['u_loc', 'g_loc', 'r_loc', 'i_loc', 'z_loc']  # Temp Hack
        dl_count = 0
        for i, galaxy in self.meta.iterrows():
            to_dl = self._get_formatted_urls(galaxy['rerun'],
                                             galaxy['run'],
                                             galaxy['camcol'],
                                             galaxy['field'])
            for band_loc, url in zip(loc_itr, to_dl):
                r = requests.get(url, stream=True)
                if r.status_code == 200:
                    file_name = os.path.join(self.data_loc, url.split('/')[-1])
                    with open(file_name, 'wb') as bz2File:
                        for chunks in r:
                            bz2File.write(chunks)
                    print('Successfully Downloaded -> ', url.split('/')[-1])
                    self.meta[band_loc] = file_name
                    dl_count += 1
                else:
                    self.meta[band_loc] = 'Failed'
                    print('Failed to download -> ', url.split('/')[-1])
        print('Number of Images Downloaded: ', dl_count)
        self.meta.to_csv(os.path.join(self.data_loc, 'updated_meta.csv'))
        print('Successfully saved the csv file with updated meta....')
        return dl_count

    def _get_formatted_urls(self, rerun, run, camcol, field):
        """For this galaxy return all possible image urls"""
        possible_bands = ['u', 'g', 'r', 'i', 'z']
        run_arr = list(str(run).strip())

        if len(run_arr) < 6:
            run_arr = ['0']*(6-len(run_arr)) + run_arr
        run_str = ''.join(run_arr)
        field_arr = list(str(field).strip())

        if len(field_arr) < 4:
            field_arr = ['0']*(4-len(field_arr)) + field_arr

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


if __name__ == "__main__":
    id = ImagesDownloader(
        url='https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/' +
            '{rerun}/{run}/{camcol}/frame-{band}-{run_str}-{camcol}-{field}.fits.bz2',
        loc='../data/images/',
        num_images=10,
        csv_loc='../data/meta_gal.csv'
    )
    num_files = id.download_fits()
