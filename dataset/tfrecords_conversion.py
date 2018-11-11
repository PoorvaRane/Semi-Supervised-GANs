#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

import numpy as np
import os
import h5py
import pdb
import pdb

# In[2]:


s3bucket_path = '/mys3bucket/tf_shards'

training = 'set_0_'
validation = 'set_1_'

cancerous = ['class_1']
non_cancerous = ['class_0', 'class_2', 'class_3']


# In[3]:


feature = {'annotation_id': tf.FixedLenFeature([], tf.int64),
           'annotation_substance_id': tf.FixedLenFeature([], tf.int64),
           'class_id' : tf.FixedLenFeature([], tf.int64),
           'image_jpg' : tf.FixedLenFeature([], tf.string),
           'slide_id' : tf.FixedLenFeature([], tf.int64),
           'weight' : tf.FixedLenFeature([], tf.float32),
           'x' : tf.FixedLenFeature([], tf.float32),
           'y' : tf.FixedLenFeature([], tf.float32),
           'zoom_factor' : tf.FixedLenFeature([], tf.int64),
          }


# In[7]:

def read_and_decode(filename_queue, label_index):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.image.decode_jpeg(features['image_jpg'])
    label = tf.convert_to_tensor(label_index, dtype=tf.uint8)
    return image, label


def create_data(s3bucket_path, mode = 'train', label = 'cancerous'):

    suffix = '_zoom_2'
    img_size = 266
    num_channels = 3
    X = []
    Y = []
    
    # Initialize data_split and class_label
    if mode == 'train':
        data_split = training
    else:
        data_split = validation
        
    if label == 'cancerous':
        class_label = cancerous
        label_index = 1
    else:
        class_label = non_cancerous
        label_index = 0

    with tf.Session() as sess:
        init_op = tf.initializers.global_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        
        for l in class_label:
            folder_name = os.path.join(s3bucket_path,data_split +  l + suffix)
            files = os.listdir(folder_name)
         
            for file in files:
                file_path = os.path.join(folder_name,file)
                print(file_path)

                #if file_path == "/mys3bucket/tf_shards/set_1_class_3_zoom_2/shard_193.tfrecords":
                #    sess.close()
                #    return X,Y
                      
                filename_queue = tf.train.string_input_producer([file_path])
                image, label = read_and_decode(filename_queue, label_index)
                image = tf.reshape(image, [img_size, img_size, num_channels])  
                threads = tf.train.start_queue_runners(sess=sess,coord=coord) 
                
                image, label = sess.run([image, label])
                image = np.array(image)
                label = np.array(label)
                X.append(image)
                Y.append(label)
            
        coord.request_stop()
        coord.join(threads)

    return X, Y 

X_cancer, Y_cancer = create_data(s3bucket_path, mode='valid', label='cancerous')
X_nocancer, Y_nocancer = create_data(s3bucket_path, mode='valid', label='noncancerous')

train_data = np.append(X_cancer, X_nocancer, 0)
train_labels = np.append(Y_cancer, Y_nocancer, 0)



f = h5py.File('valid_dataset.hdf5','w')    
grp=f.create_group('data')
adict=dict(X=train_data,Y=train_labels)
for k,v in adict.items():
    grp.create_dataset(k,data=v)
f.close()
print("Done!")


