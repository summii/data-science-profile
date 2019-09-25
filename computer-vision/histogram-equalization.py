from PIL import Image
from pylab import *
from numpy import *
import os

im = array(Image.open("himani.jpg").convert('L'))



def histeq(im, nbr_bins=256):
	imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)

	cdf = imhist.cumsum() # cumulative distribution function

	cdf = 255 * cdf/cdf[-1]  #normalize

	im2 = interp(im.flatten(), bins[:-1], cdf)

	return im2.reshape(im.shape), cdf
