import unittest
from image_getter import ImagesDownloader
import shutil
import os
import pandas as pd


class TestImageDownloader(unittest.TestCase):
    """Test Few Methods of Image Downloader"""
    def setUp(self):
        self.downloader = ImagesDownloader(
            url='https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/' +
                '{rerun}/{run}/{camcol}/frame-{band}-{run_str}-{camcol}-{field}.fits.bz2',
            loc='/tmp/images/',
            num_images=5,
            csv_loc='../data/meta_gal.csv'
        )

    def tearDown(self):
        shutil.rmtree(self.downloader.data_loc)

    def test_download_fits(self):
        num_files_downloaded = self.downloader.download_fits()
        self.assertEqual(num_files_downloaded, 25)
        metadata_path = os.path.join(self.downloader.data_loc,
                                     'updated_meta.csv')
        self.assertEqual(metadata_path, '/tmp/images/updated_meta.csv')
        self.assertTrue(os.path.isfile(metadata_path))


if __name__ == "__main__":
    unittest.main()
