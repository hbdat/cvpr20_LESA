# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:59:52 2018

@author: badat
"""

import tensorflow as tf
import pdb
import numpy as np
from sklearn.preprocessing import normalize
from tensorflow.contrib import slim

class AttentionClassifier:
    def __init__(self, vecs,unseen_vecs=None, is_residual=True, T = 10, global_pool='mean', trainable_vecs = True, dim_feature=[196, 512],
                 dim_w2v=300, n_unseen_classes=2594, lamb_factor_w2v = 0, lamb_att_span=0.00,lamb_att_global=0.00,lamb_att_dist=0.0, 
                 margin_att_global =0.0,lamb_w2v_diff=0.0,is_regularizer = False,is_batchnorm=True, is_separate_W_1=False):
        """
        Args:
            
        """
        self.lamb_att_span = lamb_att_span
        self.lamb_att_global = lamb_att_global
        self.lamb_att_dist = lamb_att_dist
        self.margin_att_global = margin_att_global
        self.lamb_factor_w2v = lamb_factor_w2v
        self.lamb_w2v_diff = lamb_w2v_diff
        self.is_separate_W_1 = is_separate_W_1
        
        self.L = dim_feature[0]
        self.D = dim_feature[1]     #### D is the feature dimension of attention windows
        self.dim_w2v = dim_w2v
        self.C = vecs.shape[0]
        self.unseen_C = n_unseen_classes
        self.T = T
        self.trainable_vecs = trainable_vecs
        self.is_residual = is_residual
        self.is_regularizer = is_regularizer
        self.is_batchnorm = is_batchnorm
        
        self.weight_initializer = tf.contrib.layers.variance_scaling_initializer()#tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.visual_labels = tf.placeholder(tf.float32, [None,self.C])
        self.ids = tf.placeholder(tf.string, [None])
        self.description = ''
        self.prefix = ''
        self.scope_sanity_counter = 0
        self.global_pool = global_pool
        
        if vecs is None:
            self.init_vecs = np.zeros((self.C,self.dim_w2v),dtype=np.float32)
        else:
            self.init_vecs = normalize(vecs)
            
        if unseen_vecs is None:
            self.unseen_vecs = np.zeros((1,self.dim_w2v),dtype=np.float32)
        else:
            self._log('constant unseen')
            self.unseen_vecs = tf.nn.l2_normalize(unseen_vecs,1)
        
    def _debug(self,p,out):
        ### Debugger ###
        def _debug_print_func(*package):
            pdb.set_trace()
            return False
        
        debug_print_op = tf.py_func(_debug_print_func, p, [tf.bool])
        
        with tf.control_dependencies(debug_print_op):
            out = tf.identity(out, name='out')
        ### Debugger ###
        return out
    
    def _feature_global(self,features):
        pool_features = None
        if self.global_pool == 'max':
            self._log('global max pooling features')
            pool_features = tf.reduce_max(features,axis=1)
        elif self.global_pool == 'mean':
            self._log('global mean pooling features')
            pool_features = tf.reduce_mean(features,axis=1)
        return pool_features
    
    def _log(self,string):
        print(self.prefix+string)
        self.description += string+'\n'
    
    def _scope_begin(self):
        self.prefix += '\t'
        self.scope_sanity_counter += 1
    
    def _scope_end(self):
        self.prefix = self.prefix[:-1]
        self.scope_sanity_counter -= 1
        
    def _augment_1(self,features,tensor_rank):
        shape = tf.shape(features)
        if tensor_rank == 3: 
            return tf.concat([features,tf.ones([shape[0],shape[1],1])],axis = 2)
        elif tensor_rank == 4:
            return tf.concat([features,tf.ones([shape[0],shape[1],shape[2],1])],axis = 3)
        else:
            return tf.concat([features,tf.ones([shape[0],1])],axis = 1)
    
    def _augment_0(self,features,tensor_rank):
        shape = tf.shape(features)
        if tensor_rank == 3: 
            return tf.concat([features,tf.zeros([shape[0],shape[1],1])],axis = 2)
        else:
            return tf.concat([features,tf.zeros([shape[0],1])],axis = 1)
    
    def _attention_layer_MLP(self,features,use_global = False, is_dropout = False):
        self._scope_begin()
        self._log('number of attention '+str(self.T))
        self._log('MLP attention model using original feature')
        with tf.variable_scope('attention_layer',reuse=tf.AUTO_REUSE):
            self.w_att_1 = tf.get_variable('w_att_1', [self.D+1,self.D], initializer=self.weight_initializer)
                        
            self._log('random init second layers')
            self.w_att_2 = tf.get_variable('w_att_2', [self.D+1,self.T], initializer=self.weight_initializer)# tf.constant(np.zeros([self.D,self.T],dtype=np.float32))#
            
            att = tf.tensordot(features,self.w_att_1,axes=[[2],[0]])                                     #[32,196,512] <== [32,196,512] tensordot [512,512]
            att_h = tf.nn.tanh(att)                                                                     #[32,196,512]
            att_h_a = self._augment_1(att_h,tensor_rank=3)                                                            #[32,196,512]
            out_att_log = tf.tensordot(att_h_a,self.w_att_2,axes=[[2],[0]])                                    #[32,196,10] <== [32,196,512] tensordot [512,10]
            out_att_log = tf.transpose(out_att_log,perm=[0,2,1])                                            #[32,10,196]
            if is_dropout:
                self.dropout_prob = tf.get_variable('dropout_prob', [], initializer=tf.constant_initializer(1e-5),trainable=False)
                self._log('dropout')
                out_att_log = tf.nn.dropout(out_att_log,keep_prob=self.dropout_prob,noise_shape=[tf.shape(out_att_log)[0],self.T,1])
            out_att = tf.nn.softmax(out_att_log,dim=2)
            f_att=tf.multiply(features[:,tf.newaxis,:,:],out_att[:,:,:,tf.newaxis])                 #hadamard product [32,10,196,512] <== [32,1,196,512] [32,10,196,1]
            f_att = tf.reduce_sum(f_att,axis=2)                                                     #[32,10,512] <== [32,10,196,512]
            self.f_global = self._feature_global(features)                                       #[32,512] <== [32,196,512]
            if use_global:
                self._log('using mean global feature')
                f_att = tf.concat([self.f_global[:,tf.newaxis,:],f_att],axis = 1)                         #[32,11,512] <== [32,1,512] [32,10,512]
            self._scope_end()
            return f_att,out_att,out_att_log
        
    def _attention_layer_MLP_separate_W_1(self,features,use_global = False, is_dropout = False):
        self._scope_begin()
        self._log('number of attention '+str(self.T))
        self._log('share attention with separate W_1')
        with tf.variable_scope('attention_layer',reuse=tf.AUTO_REUSE):
            self.w_att_1 = tf.get_variable('w_att_1', [self.T,self.D+1,self.D], initializer=self.weight_initializer)
                        
            self._log('random init second layers')
            self.w_att_2 = tf.get_variable('w_att_2', [self.D+1,self.T], initializer=self.weight_initializer)# tf.constant(np.zeros([self.D,self.T],dtype=np.float32))#
            
            att = tf.einsum("brf,tfh->btrh",features,self.w_att_1)
            att_h = tf.nn.tanh(att)                                                                     #[32,196,512]
            att_h_a = self._augment_1(att_h,tensor_rank=4)                                                            #[32,196,512]
            out_att_log = tf.einsum("btrh,ht->btr",att_h_a,self.w_att_2)
            if is_dropout:
                self.dropout_prob = tf.get_variable('dropout_prob', [], initializer=tf.constant_initializer(1e-5),trainable=False)
                self._log('dropout')
                out_att_log = tf.nn.dropout(out_att_log,keep_prob=self.dropout_prob,noise_shape=[tf.shape(out_att_log)[0],self.T,1])
            out_att = tf.nn.softmax(out_att_log,dim=2)
            f_att=tf.multiply(features[:,tf.newaxis,:,:],out_att[:,:,:,tf.newaxis])                 #hadamard product [32,10,196,512] <== [32,1,196,512] [32,10,196,1]
            f_att = tf.reduce_sum(f_att,axis=2)                                                     #[32,10,512] <== [32,10,196,512]
            self.f_global = self._feature_global(features)                                       #[32,512] <== [32,196,512]
            if use_global:
                self._log('using mean global feature')
                f_att = tf.concat([self.f_global[:,tf.newaxis,:],f_att],axis = 1)                         #[32,11,512] <== [32,1,512] [32,10,512]
            self._scope_end()
            return f_att,out_att,out_att_log
        
    
    def _attention_ortho_span_regularizer(self,out_att_log):                                    #[32,10,196]
        self._scope_begin()
        with tf.variable_scope('attention_ortho_span_regularizer',reuse=tf.AUTO_REUSE):
            self._log('attention_ortho_span_regularizer')
            out_att_norm = tf.nn.l2_normalize(out_att_log,2)                                            #[32,10,196]
            regularizer = tf.multiply(out_att_norm[:,tf.newaxis,:,:],out_att_norm[:,:,tf.newaxis,:])    #[32,10,10,196] <== [32,1,10,196] hadamard [32,10,1,196]
            regularizer = tf.reduce_sum(regularizer,axis = 3)                                           #[32,10,10] <== [32,10,10,196]
            self._log('zero-out diag')
            diag = tf.zeros((tf.shape(regularizer)[:-1]))
            regularizer = tf.linalg.set_diag(regularizer,diag)
            regularizer = tf.norm(regularizer)                                              #Frobenius norm
            self._scope_end()
            return regularizer
    
    def _attention_distribution_soft_cardinality(self):
        self._scope_begin()       
        eps = 1e-8                
        with tf.variable_scope('attention_distribution_soft_cardinality',reuse=tf.AUTO_REUSE):
            self._log('!!!!!!! attention_distribution_soft_cardinality !!!!!!!')                                 #[32,5000] <== [32,10,5000]
            self._log('!!!!!!! NO cardinality normalization !!!!!!!')                                 #[32,5000] <== [32,10,5000]
            
            mask = tf.clip_by_value(self.visual_labels,0,1)                                      #[32,5000]
            n_pos=1.0
            mask = tf.multiply(mask,1.0/n_pos)[:,tf.newaxis,:]                                  #[32,1,5000]
            P = tf.nn.softmax(self.att_logits,axis = 1)              #[32,10,5000]
            P = tf.multiply(P,mask)                   #[32,10,5000]
            P = tf.reduce_sum(P,axis=2)                  #[32,10]<==[32,10,5000]
            self._log("{} -- except shape (?,{})".format(P, self.T))
            
            P = tf.square(P)
            P = tf.reduce_mean(P,axis=1)
            regularizer = tf.reduce_mean(P)
            self._scope_end()
            return regularizer
    
    def _attention_global_regularizer(self):
        self._scope_begin()
        with tf.variable_scope('attention_span_global',reuse=tf.AUTO_REUSE):
            self._log('attention_span_global margin {}'.format(self.margin_att_global))
            self.logits_global = tf.matmul(self.f_global,tf.transpose(self.classifiers))
            loss = -tf.multiply(self.logits - self.logits_global,self.visual_labels)
            loss = tf.maximum(loss+self.margin_att_global,0)
            loss = tf.reduce_sum(loss)#/tf.reduce_sum(tf.abs(self.visual_labels))
            self._scope_end()
            return loss
    
    
    def _ranking_loss(self,logits,labels):
        eps = 1e-8
        with tf.variable_scope('ranking_loss'):
            subset_idx = tf.reduce_sum(tf.abs(labels),axis=0)
            subset_idx = tf.reshape(tf.where(subset_idx>0),[-1])
            self._log('subset labels')
            sub_labels = tf.gather(labels,subset_idx,axis=1)
            sub_logits = tf.gather(logits,subset_idx,axis=1)
            
            positive_tags=tf.clip_by_value(sub_labels,0.,1.)
            negative_tags=tf.clip_by_value(-sub_labels,0.,1.)
            mask = tf.multiply(positive_tags[:,tf.newaxis,:],negative_tags[:,:,tf.newaxis])
            pos_socre_mat=tf.multiply(sub_logits,positive_tags)
            neg_socre_mat=tf.multiply(sub_logits,negative_tags)
            
            IW_pos3=pos_socre_mat[:,tf.newaxis,:] #(Input_shape[0],1,Input_shape[1]))
            IW_neg3=neg_socre_mat[:,:,tf.newaxis] #(Input_shape[0],Input_shape[1],1))
            
            O=1+IW_neg3-IW_pos3
            O_mask = tf.multiply(mask,O)
            diff = tf.maximum(O_mask, 0)
            violation = tf.sign(diff)
            violation = tf.reduce_sum(violation,axis=1)
            violation = tf.reduce_sum(violation,axis=1)
            diff = tf.reduce_sum(diff, axis=1)
            diff = tf.reduce_sum(diff, axis=1)
            loss =  tf.reduce_mean( tf.divide(diff,violation+eps) )
            return loss
    
    def _predict_frac(self,f_att,vecs):
        with tf.variable_scope('prediction_frac',reuse =  tf.AUTO_REUSE):
            self._scope_begin()
            self._log('prediction_frac')
            self.W = tf.get_variable('W', [self.D,self.dim_w2v], initializer=self.weight_initializer)     #[512,300]
            classifiers = tf.transpose(tf.matmul(self.W,tf.transpose(vecs)))                                                        #[5000,512] <=== ([512,300] matmul [5000,300]^T)^T
            classifiers=self._augment_0(classifiers,tensor_rank=2)
            att_logits = tf.tensordot(f_att,tf.transpose(classifiers),axes = [[2],[0]])                                             #[32,10,5000] <=== [32,10,512] tensordot [512,5000]
            
            logits = tf.reduce_max(att_logits,axis = 1)
            self._scope_end()
            return logits,att_logits,classifiers
    
    def _conv(self,features):
        with tf.variable_scope('conv',reuse =  tf.AUTO_REUSE):
            l2_scale = 0.0005
            regularizer = tf.contrib.layers.l2_regularizer(l2_scale)
            self._log('l2_scale {}'.format(l2_scale))
            features = tf.reshape(features,[-1,14,14,512])
            
            if self.is_batchnorm:
                self._log('BatchNorm for conv')
                self.is_train = tf.Variable(True,trainable=False,name='is_train')
                features = slim.conv2d(features, 512, [2, 2],
                                       normalizer_fn=tf.layers.batch_normalization,
                                       normalizer_params={"training": self.is_train}
                                       ,padding='SAME', scope='conv',weights_regularizer = regularizer)
            else:
                features = slim.conv2d(features, 512, [2, 2], padding='SAME', scope='conv',weights_regularizer = regularizer)
            
            
            features = tf.reshape(features,[-1,196,512])
            print(features)
            return features

    
    def build_model_rank(self,is_conv = False,constraint_vecs = True):
        self._log('-'*30)
        self._log('build_model_rank_sum')
        
        
        if self.trainable_vecs:
            if constraint_vecs:
                self._log('trainable_vec with normalization layer')
                vecs = tf.Variable(self.init_vecs,name='vecs')
                self.vecs = tf.nn.l2_normalize(vecs,1)
            else:
                self._log('trainable_vec')
                vecs = tf.Variable(self.init_vecs,name='vecs')
                self.vecs = vecs
        else:
            self._log('const_vec')
            self.vecs = tf.constant(self.init_vecs)
        
        
        if is_conv:
            self._log('learn conv')
            features = self._conv(self.features)
        else:
            self._log('no conv')
            features = self.features
        
        features_a = self._augment_1(features,tensor_rank=3)
        if self.is_separate_W_1:
            self.f_att,self.alphas,out_att_log = self._attention_layer_MLP_separate_W_1(features_a,use_global=False,is_dropout=False)
        else:
            self.f_att,self.alphas,out_att_log = self._attention_layer_MLP(features_a,use_global=False,is_dropout=False)
        
        self.logits,self.att_logits,self.classifiers = self._predict_frac(self.f_att,self.vecs)
        self.zs_logits,self.zs_att_logits,self.zs_classifiers = self._predict_frac(self.f_att,self.unseen_vecs)
        vecs_all = tf.concat([self.unseen_vecs,self.vecs],axis=0)
        self.gzs_logits,self.gzs_att_logits,self.gzs_classifiers = self._predict_frac(self.f_att,vecs_all)
        
        self.loss_rank = self._ranking_loss(self.logits,self.visual_labels)
        self.loss_att_dist = self._attention_distribution_soft_cardinality()
        self.loss_att_span = self._attention_ortho_span_regularizer(out_att_log)
        self.loss_att_global = self._attention_global_regularizer()
        self.loss_regularizer = tf.losses.get_regularization_loss()
        self.loss_w2v_diff = tf.square(tf.norm(self.init_vecs - self.vecs))
        
        self._log('lambda att span: '+str(self.lamb_att_span))
        self._log('lambda att global: '+str(self.lamb_att_global))
        self._log('lambda att dist: '+str(self.lamb_att_dist))
        self._log('lambda w2v diff: '+str(self.lamb_w2v_diff))
        self.loss = self.loss_rank \
                        +self.lamb_att_dist*self.loss_att_dist \
                        +self.lamb_att_global*self.loss_att_global \
                        +self.lamb_att_span*self.loss_att_span \
                        +self.lamb_w2v_diff*self.loss_w2v_diff
        if self.is_regularizer and is_conv:
            self._log('use l2 regularizer')
            self.loss += self.loss_regularizer
            
        self._log('-'*30)
   