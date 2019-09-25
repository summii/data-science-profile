import os
from PIL import Image
from pylab import *
from numpy import *

im = array(Image.open("himani.jpg").convert('L'))
gray()

# create a list of filenames of all images in a folder
def get_imlist(path):
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]



def histeq(im, nbr_bins=256):
	imhist, bins = histogram(im.flatten(), nbr_bins, density=True)

	cdf = imhist.cumsum() # cumulative distribution function

	cdf = 255 * cdf/cdf[-1]  #normalize

	im2 = interp(im.flatten(), bins[:-1], cdf)

	return im2.reshape(im.shape), cdf

im2, cdf = histeq(im)
title('after')
imshow(im2)


show()