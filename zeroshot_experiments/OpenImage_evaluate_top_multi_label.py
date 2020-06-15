# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:32:44 2019

@author: badat
"""

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd) 
print('-'*30)
print(os.getcwd())
print('-'*30)
import tensorflow as tf
import time
import os
import numpy as np
import pdb
import pickle
import pandas as pd
from core.CONSE import CONSE
from tensorflow.contrib import slim
from core.model_share_attention import AttentionClassifier
from core.utils import visualize_attention,Logger,get_compress_type,LearningRate,evaluate_k,evaluate,compute_AP,\
                        evaluate_zs_df_OpenImage
from global_setting import dim_feature,train_img_path,val_img_path,n_iters,\
                                batch_size,NFS_path,description,init_w2v,test_img_path
#%%
idx_GPU=6
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
learning_rate_phase_1 = 0.001
#%%
is_save = True
path=NFS_path+'data/2018_04/'
seen_labelmap_path=path+'classes-trainable.txt'
unseen_labelmap_path=path+'unseen_labels.pkl'
dict_path=path+'class-descriptions.csv'
n_report = 50
name='eval_top_obs_2_400_mll_zs_OpenImage_log_GPU_{}_{}'.format(idx_GPU,str(time.time()).replace('.','d'))
evaluation_path = NFS_path+'results/mll_zs_OpenImage_log_GPU_5_1551726076d6135786'
save_path = NFS_path+'results/'+name
k = 5
k_zs = [3,5]
k_gzs = [10,20]
path_top_unseen = './data/2018_04/top_400_unseen.csv'
print('n_iters',n_iters)
print('compute mll and zs results!!!')
#%%
with open(init_w2v,'rb') as infile:
    seen_vecs,unseen_vecs = pickle.load(infile)
    
df_top_unseen = pd.read_csv(path_top_unseen,header=None)
idx_top_unseen = df_top_unseen.values[:,0]
assert len(idx_top_unseen) == 400
unseen_vecs = unseen_vecs[idx_top_unseen]
#%%
def parser(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'feature': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}

    parsed = tf.parse_single_example(record, feature)
    img_id = parsed['img_id']
    feature = tf.reshape(tf.decode_raw( parsed['feature'],tf.float32),dim_feature)
    label = tf.decode_raw( parsed['label'],tf.int32)
    return img_id,feature,label

def parser_test(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'feature': tf.FixedLenFeature([], tf.string),
               'seen_label': tf.FixedLenFeature([], tf.string),
               'unseen_label': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id = parsed['img_id']
    feature = tf.reshape(tf.decode_raw( parsed['feature'],tf.float32),dim_feature)
    seen_label = tf.decode_raw( parsed['seen_label'],tf.int32)
    unseen_label = tf.decode_raw( parsed['unseen_label'],tf.int32)
    unseen_label = tf.gather(unseen_label,idx_top_unseen)
    gzs_label = tf.concat([unseen_label,seen_label],axis=0)
    return img_id,feature,seen_label,unseen_label,gzs_label

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
#%%
print('Getting training classes')
seen_labelmap,unseen_labelmap, label_dict = LoadLabelMap(seen_labelmap_path, unseen_labelmap_path, dict_path)
n_seen_classes = len(seen_labelmap)
n_unseen_classes = len(unseen_labelmap)
unseen_classes = []
for idx_c in range(n_unseen_classes):
    unseen_classes.append(label_dict[unseen_labelmap[idx_c]])
unseen_classes = np.array(unseen_classes)[idx_top_unseen]

seen_classes = []
for idx_c in range(n_seen_classes):
    seen_classes.append(label_dict[seen_labelmap[idx_c]])
seen_classes = np.array(seen_classes)

classes = np.concatenate([seen_classes,unseen_classes])
#%%
dataset_test = tf.data.TFRecordDataset(test_img_path,compression_type=get_compress_type(test_img_path))
dataset_test = dataset_test.map(parser_test)
dataset_test = dataset_test.batch(100)
iterator_test = dataset_test.make_initializable_iterator()
(img_ids_test,features_test,seen_labels_test,unseen_labels_test,gzs_labels_test) = iterator_test.get_next()
#%% model
with tf.variable_scope(tf.get_variable_scope()):
    model = AttentionClassifier(vecs = seen_vecs,unseen_vecs=unseen_vecs,T=10,trainable_vecs=False,lamb_att_dist=0.1,lamb_att_global=0.001,lamb_att_span=0.01,dim_feature=dim_feature,is_batchnorm=True)
    model._log('lr {}'.format(learning_rate_phase_1))
    model._log('no shuffle')
    model._log('n_iters {}'.format(n_iters))
    model._log('adaptive learning rate')
    model._log(description)
    model._log(train_img_path)
    model._log('evaluation path {}'.format(evaluation_path))
    model.build_model_rank(is_conv=True)
#%%
if is_save:
    os.makedirs(save_path)
    os.makedirs(save_path+'/plots')
#    summary_writer = tf.summary.FileWriter(save_path, graph=tf.get_default_graph())
    with open(save_path+'/description.txt','w') as f:
        f.write(model.description)
#%%
sess = tf.InteractiveSession()
#%%
tf.global_variables_initializer().run()
saver = tf.train.Saver()

tensors_zs = [img_ids_test,features_test,unseen_labels_test]
tensors_gzs = [img_ids_test,features_test,gzs_labels_test]

def evaluate_df():
    ap_tst,predictions_tst_v,labels_tst_v = evaluate(iterator_test,[img_ids_test,features_test,seen_labels_test],model.features,model.logits,sess,model)
    print('mAP',np.mean(ap_tst))
    norm_b = np.linalg.norm(predictions_tst_v)
    F1_3_tst,P_3_tst,R_3_tst=evaluate_k(3,iterator_test,[img_ids_test,features_test,seen_labels_test],model.features,model.logits,sess,model,predictions_tst_v,labels_tst_v)
    F1_5_tst,P_5_tst,R_5_tst=evaluate_k(5,iterator_test,[img_ids_test,features_test,seen_labels_test],model.features,model.logits,sess,model,predictions_tst_v,labels_tst_v)
    print('sanity check {}'.format(np.linalg.norm(predictions_tst_v)-norm_b))
    ## reload best model
    print(np.mean(F1_3_tst),np.mean(P_3_tst),np.mean(R_3_tst))
    print(np.mean(F1_5_tst),np.mean(P_5_tst),np.mean(R_5_tst))
    df = pd.DataFrame()
    df['classes']=seen_classes
    df['F1_10']=F1_3_tst
    df['P_10']=P_3_tst
    df['R_10']=R_3_tst
    
    df['F1_20']=F1_5_tst
    df['P_20']=P_5_tst
    df['R_20']=R_5_tst
    df['ap'] = ap_tst
    return df

print('ES model')
saver.restore(sess, evaluation_path+'/model_ES.ckpt')
df_ES=evaluate_df()
df_ES_zs,df_ES_gzs=evaluate_zs_df_OpenImage(iterator_tst=iterator_test,tensors_zs=tensors_zs,tensors_gzs=tensors_gzs,
                         unseen_classes=unseen_classes,classes=classes,
                         sess=sess,model=model,k_zs = k_zs,k_gzs = k_gzs)

print('final model')
saver.restore(sess, evaluation_path+'/model.ckpt')
df_f=evaluate_df()
df_f_zs,df_f_gzs=evaluate_zs_df_OpenImage(iterator_tst=iterator_test,tensors_zs=tensors_zs,tensors_gzs=tensors_gzs,
                         unseen_classes=unseen_classes,classes=classes,
                         sess=sess,model=model,k_zs = k_zs,k_gzs = k_gzs)





if is_save:
    df_f_zs.to_csv(save_path+'/F1_f_zs_test.csv')
    df_f_gzs.to_csv(save_path+'/F1_f_gzs_test.csv')
    
    df_ES_zs.to_csv(save_path+'/F1_ES_zs_test.csv')
    df_ES_gzs.to_csv(save_path+'/F1_ES_gzs_test.csv')
    df_f.to_csv(save_path+'/F1_f_test.csv')
    df_ES.to_csv(save_path+'/F1_ES_test.csv')
#%%
sess.close()

