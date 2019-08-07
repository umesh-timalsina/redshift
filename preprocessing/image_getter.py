#! /usr/bin/env python
"""Set of Utility Methods written for getting images
    before we feed it to the swarp tool.
    1. Note, there are close to 550000 galaxy objects returned from the images,
       So, for our case, we want to specify what percentage of the images to download (of course) randomly
    2. After we download the image we want to be able to put it in a common directory where it is processed 
    3. By default this coud will download FITS For 100,000 galaxies. 
    4. That's it
"""
import numpy as np
from scrapy import Selector
import requests
import os


class ImagesDownloader():
    def __init__(self, url, loc, seed=32):
        """A class with all the methods to download fits"""
        self.url = url
        if not (os.path.exists(loc) and os.path.isdir(loc)):
            os.makedirs(loc)
        self.data_loc = loc 
        np.random.seed(seed)  # Change this for a different set of files
        
    def download_fits(self, ra, dec, specObjID):
        """Given RA and DEC for the galaxy,
        download the available fits in the sdss archive server.
        Save the file to this (self.loc) location.
        """



if __name__ == "__main__":
    id = ImagesDownloader()
    

	 
