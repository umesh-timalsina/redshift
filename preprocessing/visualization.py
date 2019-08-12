from astropy.visualization import make_lupton_rgb
from astropy.io import fits
# from astropy.utils.data import get_file_contents
import os
from matplotlib import pyplot as plt
os.system('bzip2 -dk ../data/images/frame-g-005087-5-0256.fits.bz2')
os.system('bzip2 -dk ../data/images/frame-r-005087-5-0256.fits.bz2'),
os.system('bzip2 -dk ../data/images/frame-i-005087-5-0256.fits.bz2')
g_name, r_name, i_name = ('../data/images/frame-g-005087-5-0256.fits',
                          '../data/images/frame-r-005087-5-0256.fits',
                          '../data/images/frame-i-005087-5-0256.fits')
g = fits.open(g_name)[0].data
r = fits.open(r_name)[0].data
i = fits.open(i_name)[0].data

rgb_default = make_lupton_rgb(i, r, g, filename='projected_rgb.jpeg')
plt.imshow(rgb_default, origin='lower')
