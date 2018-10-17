import os
import random
from openslide import open_slide, ImageSlide
import scipy.io as sio
import pdb
import numpy as np

slide_path = '/home/ubuntu/data/TCGA_LUSC'
slides_list = os.listdir(slide_path)
mask_path = '/home/ubuntu/data/tissue-masks-CMU/'
tar_path = '/home/ubuntu/data/patches'
#randomly sample 100 slides

print("Randomly Sampling 100 slides")
#samples = random.sample(slides_list,100)
samples = slides_list[:50]

for slide in samples :
	slide_ID = slide.split('_')[0]
	
	slide_dest = os.path.join(tar_path,slide_ID)
	if not os.path.exists(slide_dest):
		os.mkdir(slide_dest)
	slide_src = os.path.join(slide_path,slide)
	print("Reading slide "+str(slide_ID)+"...")
	image = open_slide(slide_src)
	csv_path = os.path.join(slide_dest,slide_ID+".csv")
	g = open(csv_path,'w')
	mask_src = os.path.join(mask_path,slide_ID+".mat")
	mask = sio.loadmat(mask_src)
	
	x_coords,y_coords = np.nonzero(mask['mask'])	
	x_coords = x_coords*4
	y_coords = y_coords*4
	nonzero_coords = list(zip(x_coords,y_coords))
	#pdb.set_trace()
	print("Randomly sampling patches ...")
	sample_coords = random.sample(nonzero_coords,100)
	for i in range(len(sample_coords)):
		patch = image.read_region((sample_coords[i][0],sample_coords[i][1]),0,(256,256))
		patch = patch.convert("RGB")
		outfile = os.path.join(slide_dest,"patch_"+str(i))
		patch.save(outfile,'JPEG')
		g.write(("patch_"+str(i)+str(sample_coords[i][0])+","+str(sample_coords[i][1])+"\n"))
	
	g.close()
	print("----------------------")
		
