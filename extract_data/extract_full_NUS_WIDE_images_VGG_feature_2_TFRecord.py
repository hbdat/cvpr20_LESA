# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:39:16 2018

@author: badat
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd) 
print('-'*30)
print(os.getcwd())
print('-'*30)
import tensorflow as tf
import pandas as pd
import os.path
import os
import numpy as np
import time
from tensorflow.contrib import slim
from preprocessing import preprocessing_factory
from nets import vgg
from global_setting import NFS_path
import pdb
#%%
data_set = 'Train' #'Test'
print('data_set {}'.format(data_set))
#nrows = None
path = NFS_path+'data/NUS_WIDE/'
net_name = 'vgg_19'
checkpoints_dir= './model/vgg_19/vgg_19.ckpt'
batch_size = 32
is_save = True
print('dataset: {}'.format(data_set))
num_parallel_calls=1
cap_file_size = 4000 # less than this it would be probably unavailable image
os.environ["CUDA_VISIBLE_DEVICES"]="5"
#%%
image_size = vgg.vgg_19.default_image_size
height = image_size
width = image_size
def PreprocessImage(image, network='vgg_19'):
      preprocessing_kwargs = {}
      preprocessing_fn = preprocessing_factory.get_preprocessing(name=network, is_training=False)
      height = image_size
      width = image_size
      image = preprocessing_fn(image, height, width, **preprocessing_kwargs)
      image.set_shape([height, width, 3])
      return image

def load_labels_81(filename,tag81):
    label_tags = []
    for tag in tag81:
        with open(filename+'Labels_{}.txt'.format(tag),"r") as file: 
            label_tag = np.array(file.read().splitlines(),dtype=np.float32)[:,np.newaxis]
            label_tags.append(label_tag)
    label_tags = np.concatenate(label_tags,axis=1)
    return label_tags

def get_seen_unseen_classes(file_tag1k,file_tag81):
    with open(file_tag1k,"r") as file: 
        tag1k = np.array(file.read().splitlines())
    with open(file_tag81,"r") as file:
        tag81 = np.array(file.read().splitlines())
    seen_cls_idx = np.array([i for i in range(len(tag1k)) if tag1k[i] not in tag81])
    unseen_cls_idx = np.array([i for i in range(len(tag1k)) if tag1k[i] in tag81])
    return seen_cls_idx,unseen_cls_idx,tag1k,tag81

def load_id_label_imgs(id_filename,data_partition,label1k_filename,label81_human_filename,tag81):
    with open(id_filename,"r") as file:
        id_imgs=file.readlines()
        id_imgs=[id_img.rstrip().replace('\\','/') for id_img in id_imgs]
        
    with open(data_partition,"r") as file:
        idxs_partition=file.readlines()
        idxs_partition = [idx.rstrip().replace('\\','/') for idx in idxs_partition]
        
    with open(label1k_filename,"r") as file:
        label1k_imgs=file.readlines()
    
    dict_img_id = {}
    for idx,id_img in enumerate(id_imgs):
        key = id_img.split('/')[-1]
        dict_img_id[key]=idx
    
    label1k_imgs = [np.array(label_img[:-2].split('\t'),dtype=np.float32) for label_img in label1k_imgs]
    label81_imgs = load_labels_81(label81_human_filename,tag81)
    return dict_img_id,idxs_partition,label1k_imgs,label81_imgs

def get_labels(img_id,dict_img_id,label81_imgs,label1k_imgs):
    idx_dict = dict_img_id[img_id]
#    label81 = row[1:].values
    label81 = 2*label81_imgs[idx_dict]-1 #The result is different between AllTags81 and labels_{}. AllTag81 is collected from flicker Tags whereas labels_{} is annotated by human.
    label1k = 2*label1k_imgs[idx_dict]-1
    label1k[unseen_cls_idx]=0
    return label81,label1k
#%%
file_tag1k = NFS_path + 'data/NUS_WIDE/NUS_WID_Tags/TagList1k.txt'
file_tag81 = NFS_path + 'data/NUS_WIDE/Concepts81.txt'
seen_cls_idx,unseen_cls_idx,tag1k,tag81=get_seen_unseen_classes(file_tag1k,file_tag81)

id_filename =  NFS_path + 'data/NUS_WIDE/ImageList/Imagelist.txt'.format(data_set)
label1k_filename = NFS_path + 'data/NUS_WIDE/NUS_WID_Tags/AllTags1k.txt'
label81_human_filename = NFS_path + 'data/NUS_WIDE/AllLabels/'
data_partition = NFS_path + 'data/NUS_WIDE/ImageList/{}Imagelist.txt'.format(data_set)
dict_img_id,idxs_partition,label1k_imgs,label81_imgs=load_id_label_imgs(id_filename,data_partition,label1k_filename,label81_human_filename,tag81)
#%%
feature_tfrecord_filename = NFS_path+'TFRecords/NUS_WIDE_'+data_set+'_full_feature_ZLIB.tfrecords'
data_path= './data/NUS_WIDE/Flickr/'			#path to imgs data

unavailable_img = open("./temps/test.jpg","rb").read()
tf_img_ids = []
counter = 0
counter_unavailable = 0
accum_unavailable_size = 0
for index,img_id in enumerate(idxs_partition):
    img_path = os.path.join(data_path,img_id)
    if index %1000==0:
        print('counter {} counter_unavailable {} accum_unavailable_size: {}'.format(counter,counter_unavailable,accum_unavailable_size))
    if  os.path.exists(img_path):
        size = os.path.getsize(img_path)
        img = open(img_path,"rb").read()
        if unavailable_img == img:
            counter_unavailable += 1
            accum_unavailable_size = accum_unavailable_size*0.8+size*0.2
        elif size <= cap_file_size:
            counter_unavailable += 1
        else:
            counter += 1
            tf_img_ids.append(img_path)
print(len(tf_img_ids))
#%%
def read_img(img_id):
    compressed_image = tf.read_file(img_id, 'rb')
    image = tf.image.decode_jpeg(compressed_image, channels=3)
    processed_image = PreprocessImage(image)
    
    return processed_image,img_id
#%%
print('loading data:done')
dataset = tf.data.Dataset.from_tensor_slices(tf_img_ids)
dataset = dataset.map(read_img,num_parallel_calls)
dataset = dataset.batch(batch_size).prefetch(batch_size)#.map(PreprocessImage)
processed_images,img_ids=dataset.make_one_shot_iterator().get_next()
#%%
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
g = tf.get_default_graph()
#%%
with g.as_default():
    if is_save:
        opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        feature_writer = tf.python_io.TFRecordWriter(feature_tfrecord_filename, options=opts)
    
    img_input_ph = tf.placeholder(dtype=tf.float32,shape=[None,height,width,3])
    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, _ = vgg.vgg_19(img_input_ph, num_classes=1000, is_training=False)
        init_fn = slim.assign_from_checkpoint_fn(checkpoints_dir,slim.get_model_variables())
        features = g.get_tensor_by_name('vgg_19/conv5/conv5_4/Relu:0')
    
    idx = 0
    init_fn(sess)
    n_error = 0
    while True:
        try:
            processed_images_v,img_ids_v=sess.run([processed_images,img_ids])
            features_v = sess.run(features,{img_input_ph:processed_images_v})
            print('batch no. {} n_error {}'.format(idx, n_error))
            for idx_s in range(features_v.shape[0]):
                feature = features_v[idx_s,:,:,:]
                feature = np.reshape(feature, [196, 512])
                img_id = img_ids_v[idx_s].decode('utf-8')
                key = img_id.split('/')[-1]
                label_81,label_1k=get_labels(key,dict_img_id,label81_imgs,label1k_imgs)
                example = tf.train.Example()
                example.features.feature["feature"].bytes_list.value.append(tf.compat.as_bytes(feature.tostring()))
                example.features.feature["img_id"].bytes_list.value.append(tf.compat.as_bytes(img_id))
                example.features.feature["label_1k"].bytes_list.value.append(tf.compat.as_bytes(label_1k.tostring()))
                example.features.feature["label_81"].bytes_list.value.append(tf.compat.as_bytes(label_81.tostring()))
                if is_save:
                    feature_writer.write(example.SerializeToString())
            idx += 1
        except tf.errors.OutOfRangeError:
            print('end')
            break
        except:
            n_error += 1
            print('Error')
            pass
    if is_save:
        feature_writer.close()
sess.close()
tf.reset_default_graph()