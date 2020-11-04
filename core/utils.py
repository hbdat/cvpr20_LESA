import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
from tensorflow.contrib import slim
from preprocessing.vgg_preprocessing import preprocess_for_display
from nets import vgg
from scipy import ndimage
import numpy as np
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score
from scipy.special import softmax
import pandas as pd
import time
import pdb

image_size = vgg.vgg_19.default_image_size
height = image_size
width = image_size

def get_compress_type(file_name):
    compression_type = ''
    if 'ZLIB' in file_name:
        compression_type = 'ZLIB'
    elif 'GZIP' in file_name:
        compression_type = 'GZIP'
    return compression_type

def count_records(file_name):
    c = 0
    opts = None
    if 'ZLIB' in file_name:
        print('compressed record')
        opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    for record in tf.python_io.tf_record_iterator(file_name,options=opts):
        c += 1
    return c

def compute_AP(predictions,labels):
    num_class = predictions.shape[1]
    ap=np.zeros(num_class)
    for idx_cls in range(num_class):
        prediction = np.squeeze(predictions[:,idx_cls])
        label = np.squeeze(labels[:,idx_cls])
        mask = np.abs(label)==1
        if np.sum(label>0)==0:
            continue
        binary_label=np.clip(label[mask],0,1)
        ap[idx_cls]=average_precision_score(binary_label,prediction[mask])#AP(prediction,label,names)
    return ap

def compute_F1(predictions,labels,mode_F1):
    if mode_F1 == 'overall':
        print('evaluation overall!! cannot decompose into classes F1 score')
        mask = predictions == 1
        TP = np.sum(labels[mask]==1)
        p = TP/np.sum(mask)
        r = TP/np.sum(labels==1)
        f1 = 2*p*r/(p+r)
        
#        p_2,r_2,f1_2=compute_F1_fast0tag(predictions,labels)
    else:
        num_class = predictions.shape[1]
        print('evaluation per classes')
        f1 = np.zeros(num_class)
        p = np.zeros(num_class)
        r  = np.zeros(num_class)
        for idx_cls in range(num_class):
            prediction = np.squeeze(predictions[:,idx_cls])
            label = np.squeeze(labels[:,idx_cls])
            if np.sum(label>0)==0:
                continue
            binary_label=np.clip(label,0,1)
            f1[idx_cls] = f1_score(binary_label,prediction)#AP(prediction,label,names)
            p[idx_cls] = precision_score(binary_label,prediction)
            r[idx_cls] = recall_score(binary_label,prediction)
    return f1,p,r

def get_labels(iterator,label,sess):
    sess.run(iterator.initializer)
    labels = []
    while True:
        try:
            labels_v = sess.run(label)
            labels.append(labels_v)
        except tf.errors.OutOfRangeError:
            print('end')
            break
    labels = np.concatenate(labels)
    return labels

def check_BN(model):
    try:
        model.is_train
    except AttributeError:
        return False
    else:
        return True

def mask_unlabeled(predictions,labels):
    const = np.min(predictions)-0.1
    m_predictions = predictions-const
    m_predictions = np.multiply(m_predictions,np.abs(labels))
    return m_predictions
    
def evaluate(iterator,tensors,features,logits,sess,model=None,predictions=None,labels=None,exclude=False):
    is_batchnorm = False
    if predictions is None:
        if model is not None and check_BN(model):
             is_batchnorm=True
             print('switch to inference model')
        if is_batchnorm:
            sess.run(model.is_train.assign(False))
        sess.run(iterator.initializer)
        predictions = []
        labels = []
        while True:
            try:
                img_ids_v,features_v,labels_v = sess.run(tensors)
                feed_dict = {features:features_v}
                logits_v = sess.run(logits, feed_dict)
                predictions.append(logits_v)
                labels.append(labels_v)
            except tf.errors.OutOfRangeError:
                print('end')
                break
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
#        ## for consistency treat missing label as negative value
#        print("!!!!!!!!! for consistency treat missing label as negative value !!!!!!!!!")
#        labels[labels==0]=-1
        
        if exclude:
            mask = np.sum(labels==1,1)>0
            
            print("Total test samples: {} Total samples with positive labels: {}".format(predictions.shape[0],np.sum(mask)))
            
            predictions = predictions[mask]
            labels = labels[mask]
        else:
            print('no exclude')
        
    else:
        print('Use precomputed prediction')
    assert predictions.shape==labels.shape,'invalid shape'
    if is_batchnorm:
        sess.run(model.is_train.assign(True))
    return compute_AP(predictions,labels),predictions,labels

def evaluate_k(k,iterator,tensors,features,logits,sess,model = None,predictions=None,labels=None,mode_F1='overall',is_exclude=False):
    tic = time.clock()
    is_batchnorm = False
    if predictions is None:
        if model is not None and check_BN(model):
             is_batchnorm=True
             print('switch to inference model')
        if is_batchnorm:
            sess.run(model.is_train.assign(False))
        sess.run(iterator.initializer)
        predictions = []
        labels = []
        while True:
            try:
                img_ids_v,features_v,labels_v = sess.run(tensors)
                feed_dict = {features:features_v}
                logits_v = sess.run(logits, feed_dict)
                predictions.append(logits_v)
                labels.append(labels_v)
            except tf.errors.OutOfRangeError:
                print('end')
                break
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
    else:
        print('Use precomputed prediction')
        predictions = predictions.copy()
        
        
    if is_exclude:
        mask = np.sum(labels==1,1)>0
        print("Total test samples: {} Total samples with positive labels: {}".format(predictions.shape[0],np.sum(mask)))
        predictions = predictions[mask]
        labels = labels[mask]
    else:
        print('no exclude')
        
#    predictions=mask_unlabeled(predictions,labels)
    ## binarize ##
    idx = np.argsort(predictions,axis = 1)
    for i in range(predictions.shape[0]):
        predictions[i][idx[i][-k:]]=1
        predictions[i][idx[i][:-k]]=0
    ## binarize ##
    
    assert predictions.shape==labels.shape,'invalid shape'
    if is_batchnorm:
        sess.run(model.is_train.assign(True))
    print('Inference time {} n_samples {}'.format(time.clock()-tic,predictions.shape[0]))
    return compute_F1(predictions,labels,mode_F1)

class Logger:
    def __init__(self,filename,cols,is_save=True):
        self.df = pd.DataFrame()
        self.cols = cols
        self.filename=filename
        self.is_save=is_save
    def add(self,values):
        self.df=self.df.append(pd.DataFrame([values],columns=self.cols),ignore_index=True)
    def save(self):
        if self.is_save:
            self.df.to_csv(self.filename)
    def get_max(self,col):
        return np.max(self.df[col])

class LearningRate:
    def __init__(self,lr,sess,signal_strength=0.3,limit_lr_scale=1e-3,decay_rate=0.8,patient=2):
        self.learning_rate = tf.Variable(lr,trainable = False,dtype=tf.float32)
        self.exp_moving_avg_old = 0
        self.exp_moving_avg_new = 0
        self.signal_strength = signal_strength
        self.limit_lr_scale = 1e-3
        self.decay_rate = 0.8
        self.patient = patient
        self.op_reset = self.learning_rate.assign(lr)
        self.limit_learning_rate = lr*limit_lr_scale
        self.m = 0
        self.sess = sess
        self.learning_rate_fh=tf.placeholder(dtype=tf.float32,shape=())
        self.op_assign_learning_rate = self.learning_rate.assign(self.learning_rate_fh)
        self.is_reset = False
        
    def reset(self):
        self.sess.run(self.op_reset)
        
    def get_lr(self):
        return self.learning_rate
    
    def decay(self):
        cur_lr = self.learning_rate.eval()
        new_lr=self.sess.run(self.op_assign_learning_rate,{self.learning_rate_fh:cur_lr*self.decay_rate})
        return new_lr
    
    def adapt(self,mAP):
        cur_lr = self.learning_rate.eval()
        new_lr = cur_lr
        if self.is_reset:
            self.exp_moving_avg_old=self.exp_moving_avg_new=mAP
            self.is_reset = False
        else:
            self.exp_moving_avg_old=self.exp_moving_avg_new
            self.exp_moving_avg_new = self.exp_moving_avg_new*(1-self.signal_strength)+mAP*self.signal_strength
            
            if self.exp_moving_avg_new<self.exp_moving_avg_old and cur_lr >= self.limit_learning_rate and self.m <= 0:
                print('Adjust learning rate')
                new_lr=self.decay()     #self.sess.run(self.op_assign_learning_rate,{self.learning_rate_fh:cur_lr*self.decay_rate})
                self.m = self.patient
            self.m -= 1
        return new_lr
    
def evaluate_zs_df_OpenImage(iterator_tst,tensors_zs,tensors_gzs,unseen_classes,classes,sess,model,k_zs = [3,5],k_gzs = [10,20],mode_F1='overall'):
    ap_tst_zs,predictions_zs,labels_zs=evaluate(iterator_tst,tensors_zs,model.features,model.zs_logits,sess,model)
    ap_tst_gzs,predictions_gzs,labels_gzs=evaluate(iterator_tst,tensors_gzs,model.features,model.gzs_logits,sess,model)
    model._log('mAP zs {}'.format(np.mean(ap_tst_zs)))
    model._log('mAP gzs {}'.format(np.mean(ap_tst_gzs)))
    
    norm_b = np.linalg.norm(predictions_zs)
    F1_tst,P_tst,R_tst=evaluate_k(k_zs[0],iterator_tst,tensors_zs,model.features,model.zs_logits,sess,model,predictions_zs,labels_zs,mode_F1=mode_F1)
    F1_p_tst,P_p_tst,R_p_tst=evaluate_k(k_zs[1],iterator_tst,tensors_zs,model.features,model.zs_logits,sess,model,predictions_zs,labels_zs,mode_F1=mode_F1)
    model._log('sanity check {}'.format(np.linalg.norm(predictions_zs)-norm_b))
    
    g_F1_tst,g_P_tst,g_R_tst=evaluate_k(k_gzs[0],iterator_tst,tensors_gzs,model.features,model.gzs_logits,sess,model,predictions_gzs,labels_gzs,mode_F1=mode_F1)
    g_F1_p_tst,g_P_p_tst,g_R_p_tst=evaluate_k(k_gzs[1],iterator_tst,tensors_gzs,model.features,model.gzs_logits,sess,model,predictions_gzs,labels_gzs,mode_F1=mode_F1)
    model._log('sanity check {}'.format(np.linalg.norm(predictions_zs)-norm_b))
    
    model._log('k={}: {} {} {}'.format(k_zs[0],np.mean(F1_tst), np.mean(P_tst), np.mean(R_tst)))
    model._log('k={}: {} {} {}'.format(k_zs[1],np.mean(F1_p_tst), np.mean(P_p_tst), np.mean(R_p_tst)))
    model._log('g_k={}: {} {} {}'.format(k_gzs[0],np.mean(g_F1_tst), np.mean(g_P_tst), np.mean(g_R_tst)))
    model._log('g_k={}: {} {} {}'.format(k_gzs[1],np.mean(g_F1_p_tst), np.mean(g_P_p_tst), np.mean(g_R_p_tst)))
        
    ## reload best model
    df_zs = pd.DataFrame()
    df_zs['classes']=unseen_classes
    df_zs['F1_{}'.format(k_zs[0])]=F1_tst
    df_zs['P_{}'.format(k_zs[0])]=P_tst
    df_zs['R_{}'.format(k_zs[0])]=R_tst
    
    df_zs['F1_{}'.format(k_zs[1])]=F1_p_tst
    df_zs['P_{}'.format(k_zs[1])]=P_p_tst
    df_zs['R_{}'.format(k_zs[1])]=R_p_tst
    
    df_zs['ap'] = ap_tst_zs
    
    df_gzs = pd.DataFrame()
    df_gzs['classes']=classes
    df_gzs['g_F1_{}'.format(k_gzs[0])]=g_F1_tst
    df_gzs['g_P_{}'.format(k_gzs[0])]=g_P_tst
    df_gzs['g_R_{}'.format(k_gzs[0])]=g_R_tst
    
    df_gzs['g_F1_{}'.format(k_gzs[1])]=g_F1_p_tst
    df_gzs['g_P_{}'.format(k_gzs[1])]=g_P_p_tst
    df_gzs['g_R_{}'.format(k_gzs[1])]=g_R_p_tst
    
    df_gzs['ap'] = ap_tst_gzs
    return df_zs,df_gzs