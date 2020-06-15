# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:28:46 2018

@author: badat
"""

print('Correct Version')

description = 'using GloVe w2v\n'
description += '-'*30 +'\n'
description += 'VGG'+'\n'
description += '-'*30 +'\n'
print(description)

dim_feature=[14*14,512]
batch_size=32

docker_path = './'
NFS_path = docker_path


dim_feature=[14*14,512]
n_epoches = 2
n_train_sample = 5501586

train_img_path = NFS_path+'TFRecords/train_feature_2018_04_ZLIB.tfrecords'
val_img_path = NFS_path+'TFRecords/validation_feature_2018_04_ZLIB.tfrecords'
test_img_path = NFS_path+'TFRecords/OI_seen_unseen_test_feature_2018_04_ZLIB.tfrecords'

############# n_iters #############
n_iters = (n_train_sample//batch_size)
if n_train_sample%batch_size != 0:
    n_iters += 1
n_iters *= n_epoches
############# n_iters #############
init_w2v = NFS_path+'wiki_contexts/OpenImage_w2v_context_window_10_glove-wiki-gigaword-300.pkl'
n_report = 150
############# learning_rate #############
learning_rate_phase_1 = 0.01
learning_rate_phase_2 = 0.001


#################### NUS_WIDE ####################
NUS_WIDE_train_img_path = NFS_path+'TFRecords/NUS_WIDE_Train_full_feature_ZLIB.tfrecords'
NUS_WIDE_val_img_path = NFS_path+'TFRecords/NUS_WIDE_Train_full_feature_ZLIB.tfrecords'
NUS_WIDE_test_img_path = NFS_path+'TFRecords/NUS_WIDE_Test_full_feature_ZLIB.tfrecords'

NUS_WIDE_n_train_sample = 80000
NUS_WIDE_init_w2v = NFS_path+'wiki_contexts/NUS_WIDE_pretrained_w2v_glove-wiki-gigaword-300'
NUS_WIDE_n_iters = 80000//batch_size*40
NUS_WIDE_zs_n_iters = 80000//batch_size*80
NUS_WIDE_signal_str = 0.3