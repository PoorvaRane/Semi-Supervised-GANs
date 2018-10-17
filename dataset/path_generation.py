import os
import random
from openslide import open_slide, ImageSlide
import scipy.io as sio
import pdb

slide_path = '/home/ubuntu/data/TCGA_LUSC'
slides_list = os.listdir(slide_path)
mask_path = '/home/ubuntu/data/tissue-masks-CMU/'
tar_path = '/home/ubuntu/data/patches'
#randomly sample 100 slides

samples = random.sample(slides_list,100)

for slide in samples : 
	slide_dest = os.path.join(tar_path,slide)
	if not os.path.exists(slide_dest):
		os.mkdir(slide_dest)
	slide_src = os.path.join(slide_path,slide)
	image = open_slide(slide_src)
	slide_ID = slide.strip('_')[0]
	mask_src = os.path.join(mask_path,slide_ID+".mat")
	mask = sio.loadmat(mask_src)
	x_coords,y_coords = np.nonzero(mask)	
	x_coords = x_coords*4
	y_coords = y_coords*4
	nonzero_coords = list(zip(x_coords,y_coords))
	sample_coords = random.sample(nonzero_coords,100)
	for i in range(len(sample_coords)):
		patch = image.read_region((sample_coords[i][0],sample_coords[i][1]),0,256)
		outfile = os.path.join(slide_dest,"path_"+i+".jpg")

		pdb.set_trace()		
