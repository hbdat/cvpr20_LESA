# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:23:35 2019

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
from global_setting_Pegasus import dim_feature,train_img_path,val_img_path,n_iters,\
                                batch_size,NFS_path,description,init_w2v,test_img_path
#%%
idx_GPU=7
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
learning_rate_phase_1 = 0.001
print('Continue training')
#%%
is_save = True
path=NFS_path+'data/2018_04/'
seen_labelmap_path=path+'classes-trainable.txt'
unseen_labelmap_path=path+'unseen_labels.pkl'
dict_path=path+'class-descriptions.csv'
n_report = 50
name='mll_tzs_OpenImage_log_GPU_{}_{}'.format(idx_GPU,str(time.time()).replace('.','d'))
save_path = NFS_path+'results/'+name
k = 5
k_zs = [3,5]
k_gzs = [10,20]
path_top_unseen = './data/2018_04/top_400_unseen.csv'
print('n_iters',n_iters)
print('compute mll and top class zs results!!!')
#%%
with open(init_w2v,'rb') as infile:
    seen_vecs,unseen_vecs = pickle.load(infile)
    
df_top_unseen = pd.read_csv(path_top_unseen,header=None)
idx_top_unseen = df_top_unseen.values[:,0]
assert len(idx_top_unseen) == 399
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
dataset_tr = tf.data.TFRecordDataset(train_img_path,compression_type=get_compress_type(train_img_path))
dataset_tr = dataset_tr.map(parser)
#dataset_tr = dataset_tr.shuffle(20000)
dataset_tr = dataset_tr.batch(batch_size)
dataset_tr = dataset_tr.repeat()
iterator_tr = dataset_tr.make_initializable_iterator()
(img_ids_tr,features_tr,labels_tr) = iterator_tr.get_next()

dataset_val = tf.data.TFRecordDataset(val_img_path,compression_type=get_compress_type(val_img_path))
dataset_val = dataset_val.map(parser)
dataset_val = dataset_val.batch(1000)
iterator_val = dataset_val.make_initializable_iterator()
(img_ids_val,features_val,labels_val) = iterator_val.get_next()

dataset_test = tf.data.TFRecordDataset(test_img_path,compression_type=get_compress_type(test_img_path))
dataset_test = dataset_test.map(parser_test)
dataset_test = dataset_test.batch(1000)
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
    model.build_model_rank(is_conv=True)
#%%
sess = tf.InteractiveSession()
#%%
model._log('adaptive learning rate')
lr = LearningRate(learning_rate_phase_1,sess)
model._log('signal_str {}'.format(lr.signal_strength))
model._log('patient {}'.format(lr.patient))
optimizer = tf.train.RMSPropOptimizer(
      lr.get_lr(),
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )
grads = optimizer.compute_gradients(model.loss)
print('-'*30)
print('Decompose update ops')
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.apply_gradients(grads)
print('-'*30)
#%%
tf.global_variables_initializer().run()
saver = tf.train.Saver()
#%%
tf.summary.scalar('batch_loss', model.loss)
for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

for grad, var in grads:
    tf.summary.histogram(var.op.name+'/gradient', grad)

summary_op = tf.summary.merge_all()
if is_save:
    os.makedirs(save_path)
    os.makedirs(save_path+'/plots')
    summary_writer = tf.summary.FileWriter(save_path, graph=tf.get_default_graph())
    with open(save_path+'/description.txt','w') as f:
        f.write(model.description)
    ##### test #####
    saver.save(sess, save_path+'/model.ckpt')
    ##### test #####
#%%
accum_l = 0
alpha = 0.9
sess.run(iterator_tr.initializer)
logger = Logger(cols=['index','l','l_rank','l_att_span','l_att_global','l_att_dist','mF1','mAP','lr'],filename=save_path+'/log.csv',
                is_save=is_save)
eval_interval = max((n_iters//n_report),500)
tic = time.clock()
tensors_zs = [img_ids_test,features_test,unseen_labels_test]
tensors_gzs = [img_ids_test,features_test,gzs_labels_test]
for i in range(n_iters):
    img_ids_tr_v,features_tr_v,labels_tr_v = sess.run([img_ids_tr,features_tr,labels_tr])

    mask = np.sum(labels_tr_v>0,1)>0
    if np.sum(mask)==0:
        continue
    feed_dict = {model.features:features_tr_v[mask],model.visual_labels:labels_tr_v[mask]}
    _, l,l_rank,l_att_span,l_att_global,l_att_dist = sess.run([train, model.loss,model.loss_rank,model.loss_att_span,model.loss_att_global,model.loss_att_dist], feed_dict)
    accum_l = l*(1-alpha)+alpha*accum_l
    if np.isnan(l):
        pdb.set_trace()
    if (i+1) % (n_iters//4)==0:
        print('-'*30)
        print('test zeroshot')
        df_f_zs,df_f_gzs=evaluate_zs_df_OpenImage(iterator_tst=iterator_test,tensors_zs=tensors_zs,tensors_gzs=tensors_gzs,
                         unseen_classes=unseen_classes,classes=classes,
                         sess=sess,model=model,k_zs = k_zs,k_gzs = k_gzs)
        print('-'*30)
    if i % eval_interval == 0 or i == n_iters-1:
        print('Time elapse: ',time.clock()-tic)
        tic = time.clock()
        ap_val,predictions_mll,labels_mll=evaluate(iterator_val,[img_ids_val,features_val,labels_val],model.features,model.logits,sess,model)
        F1_val,P_val,R_val=evaluate_k(k,iterator_val,[img_ids_val,features_val,labels_val],model.features,model.logits,sess,model,predictions_mll,labels_mll)
        
        mF1_val,mP_val,mR_val,mAP_val = [np.mean(F1_val),np.mean(P_val),np.mean(R_val),np.mean(ap_val)]
        learning_rate=lr.adapt(mAP_val)
        values = [i,l,l_rank,l_att_span,l_att_global,l_att_dist ,mF1_val,mAP_val,learning_rate]

        logger.add(values)
        print('{} loss: {} l_rank: {} l_att_span: {} l_att_global: {} l_att_dist: {} mF1: {} mAP: {} lr: {}'.format(*values))
        print('Precision: {} Recall: {}'.format(mP_val,mR_val))
        logger.save()

        print('learning rate',learning_rate)
        if is_save and mAP_val >= logger.get_max('mAP'):
            saver.save(sess, save_path+'/model_ES.ckpt')
            

def evaluate_df():
    ap_tst,predictions_mll,labels_mll = evaluate(iterator_test,[img_ids_test,features_test,seen_labels_test],model.features,model.logits,sess,model)
    F1_3_tst,P_3_tst,R_3_tst=evaluate_k(3,iterator_test,[img_ids_test,features_test,seen_labels_test],model.features,model.logits,sess,model,predictions_mll,labels_mll)
    F1_5_tst,P_5_tst,R_5_tst=evaluate_k(5,iterator_test,[img_ids_test,features_test,seen_labels_test],model.features,model.logits,sess,model,predictions_mll,labels_mll)
    
    ## reload best model
    print('mAP',np.mean(ap_tst))
    print('k=3',np.mean(F1_3_tst),np.mean(P_3_tst),np.mean(R_3_tst))
    print('k=5',np.mean(F1_5_tst),np.mean(P_5_tst),np.mean(R_5_tst))
    df = pd.DataFrame()
    df['classes']=seen_classes
    df['F1_3']=F1_3_tst
    df['P_3']=P_3_tst
    df['R_3']=R_3_tst
    
    df['F1_5']=F1_5_tst
    df['P_5']=P_5_tst
    df['R_5']=R_5_tst
    df['ap'] = ap_tst
    return df


print('final model')
if is_save:
    saver.save(sess, save_path+'/model.ckpt')
df_f=evaluate_df()
df_f_zs,df_f_gzs=evaluate_zs_df_OpenImage(iterator_tst=iterator_test,tensors_zs=tensors_zs,tensors_gzs=tensors_gzs,
                         unseen_classes=unseen_classes,classes=classes,
                         sess=sess,model=model,k_zs = k_zs,k_gzs = k_gzs)

print('ES model')
saver.restore(sess, save_path+'/model_ES.ckpt')
df_ES=evaluate_df()
df_ES_zs,df_ES_gzs=evaluate_zs_df_OpenImage(iterator_tst=iterator_test,tensors_zs=tensors_zs,tensors_gzs=tensors_gzs,
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

