#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
import scipy.ndimage
import numpy as np
import matplotlib.path as mpltPath
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import shuffle
from openslide import open_slide, ImageSlide
import scipy.io as sio
import pdb
import sys
import threading


# In[ ]:


#File paths
slide_path = '/mys3bucket/TCGA_LUSC'
slides = os.listdir(slide_path)
save_path = '/mys3bucket/patch_data'

no_patches = 100
chunk_size = 20
img_size = 256


# In[ ]:


def split(data):
    N = len(data)
    trn_idx = int(np.ceil(0.8*N))
    train = data[:trn_idx]
    test = data[trn_idx:]
    
    return train,test


# In[ ]:


def get_mask(coords):
    # Construct polygon
    p = Path(coords)
    #plot = coords
    #plot.append(coords[0])
    #xs, ys = zip(*plot)
    #plt.figure()
    #plt.plot(xs,ys)
    # Get min and max
    coords.sort(key=lambda x: x[0],reverse=True)
    xmin,xmax = coords[-1][0],coords[0][0]
    coords.sort(key=lambda x: x[1],reverse=True)
    ymin,ymax = coords[-1][1],coords[0][1]
    maximum = max(xmax,ymax)
    minimum = min(xmin,ymin)
    
    # Create (c_x, c_y) using meshgrid
    #print("Done generating meshgrid!")
    step = 10
    x, y = np.mgrid[minimum:maximum:step, minimum:maximum:step]
    #print("Done generating meshgrid!")
    #print("Flattening")
    x, y = x.flatten(), y.flatten()
    #print("Vstacking")
    center_points = np.vstack((x,y)).T
    #print("Checking if polygon contains points")
    #pdb.set_trace()
    center_grid = p.contains_points(center_points)
    #center_mask = center_grid.reshape((maximum-minimum),(maximum-minimum)).astype(int) 
    # Randomly sample points from mask
    #print("Getting all the ones from the grid")
    in_points = np.nonzero(center_grid)
    x_coords, y_coords = x[in_points], y[in_points]
    min_len = min(no_patches, y_coords.shape[0])
    #sample min_len coordinates from the mask
    sample_idxs = np.random.choice(np.arange(len(x_coords)), min_len)
    #plt.scatter(x_coords[sample_idxs], y_coords[sample_idxs])
    #plt.show()
    return x_coords[sample_idxs],y_coords[sample_idxs]


# In[ ]:


def read_patches(x_coords,y_coords,slide_src,label):
    gen_dataX = []
    gen_dataY = []
    image = open_slide(slide_src)
    for i in range(len(x_coords)):
        #print("Reading patch",i)
        patch = image.read_region((x_coords[i]-(img_size//2),y_coords[i]-(img_size//2)),0,(256,256)) #find top left pixel
        patch = patch.convert("RGB")
        #Code to save patches as images
        outfile = "patch_"+str(i)+".jpg"
        #patch.save(outfile,'JPEG')
        patch = np.array(patch)
        # check for black patches
        if not np.sum(patch)==0 :
            gen_dataX.append(patch)
            gen_dataY.append(label)
        
        #g.write(("patch_"+str(count)+","+str(x_coords[i])+","+str(y_coords[i])+"\n"))
    
    image.close()
    print("Generated patches!")
    return gen_dataX,gen_dataY


# In[ ]:


def get_slide_path(slideID):
    for slide in slides:
        if str(slideID) == str(slide.split('_')[0]):
            return os.path.join(slide_path,slide)
    return -1


# In[ ]:


def get_random_polygon(shape):
    if len(shape)>1:
        return shape
    return -1


# In[ ]:


def generate_data(data,mode,ltype):
       
    for slide in data:
        DATAX = []
        DATAY = [] 
        count = 0
        slide_src = get_slide_path(slide)
        print(str(slide)+" has "+ str(len(data[slide]))+" annotations")
        for polygon in data[slide]:
            count+=1
            coords = [tuple(x) for x in polygon]
            x_coords,y_coords = get_mask(coords)
            # Get label
            if ltype == 'cancer':
                label = 1
            else:
                label = 0

            X,Y = read_patches(x_coords,y_coords,slide_src,label)
            print(len(Y))
            DATAX.extend(X)
            DATAY.extend(Y)

            print(">>>>"+str(count))

            #Saving chunks of data containing slide_threshold*no_patches
        #pdb.set_trace()
        outfile = os.path.join(save_path,mode,"c_"+str(slide))
        try:
           np.savez(outfile,np.asarray(DATAX),np.asarray(DATAY))
        except:
          print("Failed for : ", slide)
        print("*****************************************************")
        


# In[ ]:


def get_statistics(data):
    count_cancerous = 0
    count_noncancerous = 0
    cancer_dict = {}
    noncancer_dict = {}
    for annotation in data:
        slide = annotation['slideId']
        slide_src = get_slide_path(slide)
        shape = annotation['shape']
        polygon = get_random_polygon(shape)
        if not slide_src == -1 and not polygon == -1 :
            if (annotation['annotationSubstanceId'] in [330,331]) :
                if slide not in cancer_dict:
                    cancer_dict[slide] = []
                    cancer_dict[slide].append(polygon)
                else:
                    cancer_dict[slide].append(polygon)
                count_cancerous+=1
            else:
                if slide not in noncancer_dict:
                    noncancer_dict[slide] = []
                    noncancer_dict[slide].append(polygon)
                else:
                    noncancer_dict[slide].append(polygon)                
                count_noncancerous+=1
    return cancer_dict, noncancer_dict
        


# In[ ]:


#Shuffle the data
f = open("/mys3bucket/Annotations/annotations.txt", encoding="utf-8")
data = json.loads(f.read())
f.close()


#shuffle(data)
train,test = split(data)
cancer, noncancer = get_statistics(train)


# In[ ]:


#Generate Cancerous Patches
threads = []
batch = 0
slide_count = 0
slide_list = os.listdir(os.path.join(save_path,'train'))
print(slide_list)
temp_dict = {}

for slide in cancer:
    if "c_"+str(slide)+".npz" not in slide_list and slide!=81063 :
    	temp_dict[slide] = cancer[slide]
    	batch+=1
    	slide_count+=1
    	if slide_count == 4:
        	 break
    	if batch == 3:
        	batch=0
        	print(len(temp_dict))
        	t = threading.Thread(target=generate_data, args=(temp_dict,'train','cancer',))
        	print("Launching thread....")
        	t.start()
        	threads.append(t)
        	temp_dict={}
    
for t in threads:
    t.join()
    
print("Train data generated!")


# In[ ]:
'''
remaining = {}
count = 0
for slide in noncancer:
	if str(slide)+".npz" not in slide_list and slide!=81063:
		remaining[slide] = noncancer[slide]
		count+=1
	if count == 38:
		break
generate_data(remaining,'train','noncancer')
#print("Dev data generated!")
'''

# In[ ]:


#generate_data(test, no_test_slides,'test')
#print("Test data generated! ")

