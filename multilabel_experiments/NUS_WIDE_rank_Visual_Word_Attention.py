# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 23:46:08 2018

@author: badat
"""
import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd) 
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import tensorflow as tf
import os
import numpy as np
import pdb
import time
import pickle
import pandas as pd
from tensorflow.contrib import slim
from core.model_share_attention import AttentionClassifier
from core.utils import visualize_attention,evaluate_k,Logger,get_compress_type,LearningRate,evaluate
from global_setting_Pegasus import dim_feature,NUS_WIDE_train_img_path,NUS_WIDE_test_img_path,NUS_WIDE_val_img_path,\
                                    NUS_WIDE_n_iters,NFS_path,batch_size,NUS_WIDE_init_w2v,description,NUS_WIDE_signal_str
#from core.model import AttentionClassifier
idx_GPU=0
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
learning_rate_phase_1 = 0.001
#%%
k=5
is_save = True
dim_w2v = 300
path=NFS_path+'data/2018_04/'
labelmap_path=path+'classes-trainable.txt'
n_report = 50
name='dist_elu_mll_NUS_WIDE_log_GPU_{}_{}'.format(idx_GPU,str(time.time()).replace('.','d'))
save_path = NFS_path+'results/'+name
print('k',k)
#%%
def parser(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'feature': tf.FixedLenFeature([], tf.string),
               'label_1k': tf.FixedLenFeature([], tf.string),
               'label_81': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id = parsed['img_id']
    feature = tf.reshape(tf.decode_raw( parsed['feature'],tf.float32),dim_feature)
    label_1k = tf.decode_raw( parsed['label_1k'],tf.float32)
    label_81 = tf.decode_raw( parsed['label_81'],tf.float32)
    return img_id,feature,label_1k,label_81
#%%
def get_seen_unseen_classes(file_tag1k,file_tag81):
    with open(file_tag1k,"r") as file: 
        tag1k = np.array(file.read().splitlines())
    with open(file_tag81,"r") as file:
        tag81 = np.array(file.read().splitlines())
    seen_cls_idx = np.array([i for i in range(len(tag1k)) if tag1k[i] not in tag81])
    unseen_cls_idx = np.array([i for i in range(len(tag1k)) if tag1k[i] in tag81])
    return seen_cls_idx,unseen_cls_idx,tag1k,tag81

file_tag1k = NFS_path + 'data/NUS_WIDE/NUS_WID_Tags/TagList1k.txt'
file_tag81 = NFS_path + 'data/NUS_WIDE/Concepts81.txt'
seen_cls_idx,unseen_cls_idx,tag1k,tag81=get_seen_unseen_classes(file_tag1k,file_tag81)
#%% load context information
with open(NUS_WIDE_init_w2v,'rb') as infile:
    vecs = pickle.load(infile)[1]
    
#%%
n_classes = tag81.shape[0]
print('n_classes',n_classes)
#%%
#print('-'*30)
#print('Warning: no shuffle training sets')
#print('-'*30)
dataset_tr = tf.data.TFRecordDataset(NUS_WIDE_train_img_path,compression_type=get_compress_type(NUS_WIDE_train_img_path))
dataset_tr = dataset_tr.map(parser)
dataset_tr = dataset_tr.skip(5000)
#dataset_tr = dataset_tr.shuffle(20000)
dataset_tr = dataset_tr.batch(batch_size)
dataset_tr = dataset_tr.repeat()
iterator_tr = dataset_tr.make_initializable_iterator()
(img_ids_tr,features_tr,labels_1k_tr,labels_81_tr) = iterator_tr.get_next()
#%%
dataset_val = tf.data.TFRecordDataset(NUS_WIDE_val_img_path,compression_type=get_compress_type(NUS_WIDE_val_img_path))
dataset_val = dataset_val.map(parser)
dataset_val = dataset_val.take(5000)
dataset_val = dataset_val.batch(1000)
iterator_val = dataset_val.make_initializable_iterator()
(img_ids_val,features_val,labels_1k_val,labels_81_val) = iterator_val.get_next()
#%%
dataset_tst = tf.data.TFRecordDataset(NUS_WIDE_test_img_path,compression_type=get_compress_type(NUS_WIDE_test_img_path))
dataset_tst = dataset_tst.map(parser)
dataset_tst = dataset_tst.batch(1000)
iterator_tst = dataset_tst.make_initializable_iterator()
(img_ids_tst,features_tst,labels_1k_tst,labels_81_tst) = iterator_tst.get_next()
#%%
with tf.variable_scope(tf.get_variable_scope()):
    model = AttentionClassifier(vecs,T=10,is_batchnorm=True,trainable_vecs=False,lamb_att_dist=0.0001,lamb_att_global=0.001,lamb_att_span=0.01,dim_feature=dim_feature)
    model._log('lr {}'.format(learning_rate_phase_1))
    model._log('n_iters {}'.format(NUS_WIDE_n_iters))
    model._log('no shuffle')
    model._log('adaptive learning rate')
    model._log(description)
    model._log('train_img: '+NUS_WIDE_train_img_path)
    model._log('test_img: '+NUS_WIDE_test_img_path)
    model._log('val_img: '+NUS_WIDE_val_img_path)
    model.build_model_rank(is_conv=True)
#_ = input('confirm??')
#%%
sess = tf.InteractiveSession()
#%%
model._log('adaptive learning rate')
lr = LearningRate(learning_rate_phase_1,sess,signal_strength=NUS_WIDE_signal_str,patient=5)
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
eval_interval = max((NUS_WIDE_n_iters//n_report),500)
tic = time.clock()
#sess.run(model.dropout_prob.assign(0.5))
for i in range(NUS_WIDE_n_iters):
    img_ids_tr_v,features_tr_v,labels_81_tr_v = sess.run([img_ids_tr,features_tr,labels_81_tr])
    feed_dict = {model.features:features_tr_v,model.visual_labels:labels_81_tr_v}
    _, l,l_rank,l_att_span,l_att_global,l_att_dist = sess.run([train, model.loss,model.loss_rank,model.loss_att_span,model.loss_att_global,model.loss_att_dist], feed_dict)
    accum_l = l*(1-alpha)+alpha*accum_l
    if np.isnan(l):
        pdb.set_trace()
#    if i%1000 == 0 and is_save:
#        summary = sess.run(summary_op, feed_dict)
#        summary_writer.add_summary(summary, i)

    if i % eval_interval == 0 or i == NUS_WIDE_n_iters-1:
        print('Time elapse: ',time.clock()-tic)
        tic = time.clock()
        F1_val,P_val,R_val=evaluate_k(k,iterator_val,[img_ids_val,features_val,labels_81_val],model.features,model.logits,sess)
        ap_val,_,_=evaluate(iterator_val,[img_ids_val,features_val,labels_81_val],model.features,model.logits,sess)
#        mean_all_ms,mean_nonzero_ms=get_mean_non_zero([F1_val,P_val,R_val])
        mF1_val,mP_val,mR_val,mAP_val = [np.mean(F1_val),np.mean(P_val),np.mean(R_val),np.mean(ap_val)]
        learning_rate=lr.adapt(mF1_val)
        values = [i,l,l_rank,l_att_span,l_att_global,l_att_dist ,mF1_val,mAP_val,learning_rate]

        logger.add(values)
        print('{} loss: {} l_rank: {} l_att_span: {} l_att_global: {} l_att_dist: {} mF1: {} mAP: {} lr: {}'.format(*values))
        print('Precision: {} Recall: {}'.format(mP_val,mR_val))
        logger.save()

        print('learning rate',learning_rate)
        if is_save and mF1_val >= logger.get_max('mF1'):
            saver.save(sess, save_path+'/model_ES.ckpt')

def evaluate_df():
    F1_3_tst,P_3_tst,R_3_tst=evaluate_k(3,iterator_tst,[img_ids_tst,features_tst,labels_81_tst],model.features,model.logits,sess,model)
    F1_5_tst,P_5_tst,R_5_tst=evaluate_k(5,iterator_tst,[img_ids_tst,features_tst,labels_81_tst],model.features,model.logits,sess,model)
    ap_tst,_,_=evaluate(iterator_tst,[img_ids_tst,features_tst,labels_81_tst],model.features,model.logits,sess,model)
    ## reload best model
    print('mAP',np.mean(ap_tst))
    print('k=3',np.mean(F1_3_tst),np.mean(P_3_tst),np.mean(R_3_tst))
    print('k=5',np.mean(F1_5_tst),np.mean(P_5_tst),np.mean(R_5_tst))
    df = pd.DataFrame()
    df['classes']=tag81
    df['F1_3']=F1_3_tst
    df['P_3']=P_3_tst
    df['R_3']=R_3_tst
    
    df['F1_5']=F1_5_tst
    df['P_5']=P_5_tst
    df['R_5']=R_5_tst
    
    df['ap'] = ap_tst
    return df

saver.save(sess, save_path+'/model.ckpt')
df_f=evaluate_df()


saver.restore(sess, save_path+'/model_ES.ckpt')
df_ES=evaluate_df()

if is_save:
    df_f.to_csv(save_path+'/F1_f_test.csv')
    df_ES.to_csv(save_path+'/F1_ES_test.csv')
#%%
#for i in range(2):
#    img_ids_tr_v,features_tr_v,labels_tr_v = sess.run([img_ids_tr,features_tr,labels_tr])
#    feed_dict = {model.features:features_tr_v,model.visual_labels:labels_tr_v}
#    alphas_v, l = sess.run([model.alphas, model.loss], feed_dict)
#    pdb.set_trace()
#    visualize_attention(img_ids_tr_v,alphas_v,sess,img_path=save_path+'/plots/')
sess.close()