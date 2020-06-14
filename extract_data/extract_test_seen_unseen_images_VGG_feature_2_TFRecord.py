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
from global_setting_Cygnus import NFS_path
import pickle
import pdb
#%%
data_set = 'test'
print(data_set)
#nrows = None
version = '2018_04'
path=NFS_path+'data/{}/'.format(version)
net_name = 'vgg_19'
checkpoints_dir= './model/vgg_19/vgg_19.ckpt'
batch_size = 64
is_save = True
print('dataset: {} version: {}'.format(data_set,version))
num_parallel_calls=1
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
#%%
print('loading data')
df_label = pd.read_csv(path+'/{}-annotations-human-imagelabels.csv'.format(data_set))
seen_labelmap_path=path+'classes-trainable.txt'
unseen_labelmap_path=path+'unseen_labels.pkl'
dict_path =  path+'class-descriptions.csv'
feature_tfrecord_filename = './TFRecords/OI_seen_unseen_'+data_set+'_feature_'+version+'_ZLIB.tfrecords'
data_path= './image_data/'+data_set+'/'#'./image/'#./image_data/val/
print('feature_tfrecord_filename',feature_tfrecord_filename)
#%% split dataframe
print('partitioning data')
capacity = 40000
partition_df = []
t = len(df_label)//capacity
for idx_cut in range(t):
    #print(idx_cut)
    partition_df.append(df_label.iloc[idx_cut*capacity:(idx_cut+1)*capacity])
partition_df.append(df_label.iloc[t*capacity:])
#%%
files=[]
partition_idxs = []
for idx_partition,partition in enumerate(partition_df):
    #print(idx_partition)
    file_partition=[ img_id+'.jpg' for img_id in partition['ImageID'].unique() if os.path.isfile(data_path+img_id+'.jpg')]
    files.extend(file_partition)
    partition_idxs.extend([idx_partition]*len(file_partition))
n_samples = len(files)
print('number of sample: {} dataset: {}'.format(n_samples,data_set))
#%%
def LoadLabelMap(seen_labelmap_path, unseen_labelmap_path, dict_path):
  seen_labelmap = [line.rstrip() for line in tf.gfile.GFile(seen_labelmap_path)]
  with open(unseen_labelmap_path,'rb') as infile:
    unseen_labelmap = pickle.load(infile).tolist()
  label_dict = {}
  for line in tf.gfile.GFile(dict_path):
    words = [word.strip(' "\n') for word in line.split(',', 1)]
    label_dict[words[0]] = words[1]

  return seen_labelmap,unseen_labelmap, label_dict

predictions_eval = 0
predictions_eval_resize = 0

seen_labelmap, unseen_labelmap, label_dict = LoadLabelMap(seen_labelmap_path, unseen_labelmap_path, dict_path)
num_seen_classes = len(seen_labelmap)
num_unseen_classes = len(unseen_labelmap)
print('num_seen_classes',num_seen_classes,'num_unseen_classes',num_unseen_classes)
#%%
def read_img(file,partition_idx):
    compressed_image = tf.read_file(data_path+file, 'rb')
    image = tf.image.decode_jpeg(compressed_image, channels=3)
    processed_image = PreprocessImage(image)
    
    return processed_image,file,partition_idx

def get_label(processed_image,file,partition_idx):
    img_id = file.decode('utf-8').split('.')[0]
    df_img_label=partition_df[partition_idx].query('ImageID=="{}"'.format(img_id))
    seen_label = np.zeros(num_seen_classes,dtype=np.int32)
    unseen_label = np.zeros(num_unseen_classes,dtype=np.int32)
    #print(len(df_img_label))
    for index, row in df_img_label.iterrows():
        if row['LabelName'] in seen_labelmap:
            idx=seen_labelmap.index(row['LabelName'])
            seen_label[idx] = 2*row['Confidence']-1
        if row['LabelName'] in unseen_labelmap:
            idx=unseen_labelmap.index(row['LabelName'])
            unseen_label[idx] = 2*row['Confidence']-1
    #print(partition_idx)
    return processed_image,img_id,seen_label,unseen_label
#%%
print('loading data:done')
dataset = tf.data.Dataset.from_tensor_slices((files,partition_idxs))
dataset = dataset.map(read_img,num_parallel_calls)
dataset = dataset.map(lambda processed_images,file,partition_idx: tuple(tf.py_func(get_label, [processed_images,file,partition_idx], [tf.float32, tf.string, tf.int32, tf.int32])),num_parallel_calls)
dataset = dataset.batch(batch_size).prefetch(batch_size)#.map(PreprocessImage)
processed_images,img_ids,seen_labels,unseen_labels=dataset.make_one_shot_iterator().get_next()
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
    while True:#idx < 3125:
        try:
            processed_images_v,img_ids_v,seen_labels_v,unseen_labels_v=sess.run([processed_images,img_ids,seen_labels,unseen_labels])
            features_v = sess.run(features,{img_input_ph:processed_images_v})
            print('batch no. {}'.format(idx))
            for idx_s in range(features_v.shape[0]):
                feature = features_v[idx_s,:,:,:]
                feature = np.reshape(feature, [196, 512])
                img_id = img_ids_v[idx_s]
                seen_label = seen_labels_v[idx_s,:]
                unseen_label = unseen_labels_v[idx_s,:]
                example = tf.train.Example()
                example.features.feature["feature"].bytes_list.value.append(tf.compat.as_bytes(feature.tostring()))
                example.features.feature["img_id"].bytes_list.value.append(tf.compat.as_bytes(img_id))
                example.features.feature["seen_label"].bytes_list.value.append(tf.compat.as_bytes(seen_label.tostring()))
                example.features.feature["unseen_label"].bytes_list.value.append(tf.compat.as_bytes(unseen_label.tostring()))
                if is_save:
                    feature_writer.write(example.SerializeToString())
                    feature_writer.flush()
                
            idx += 1
        except tf.errors.OutOfRangeError:
            print('end')
            break
    if is_save:
        feature_writer.close()
sess.close()
tf.reset_default_graph()