# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:40:20 2021

@author: k_mat
"""
import os
import sys

import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv2D, Conv2DTranspose, BatchNormalization, Activation, GlobalAveragePooling2D, Lambda, Input, Concatenate, Add, UpSampling2D, LeakyReLU, ZeroPadding2D,Multiply, DepthwiseConv2D, MaxPooling2D, LayerNormalization
from tensorflow.keras.models import Model

from .efficientnetv2 import effnetv2_model

sys.path.append('../')
WEIGHT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"weights/")
USE_TPU = False

if USE_TPU:
    batch_norm = tf.keras.layers.experimental.SyncBatchNormalization
else:
    batch_norm = BatchNormalization

def cbr(x, out_layer, kernel, stride, name, bias=False, use_batchnorm=True):
    x = Conv2D(out_layer, kernel_size=kernel, strides=stride,use_bias=bias, padding="same", name=name+"_conv")(x)
    if use_batchnorm:
        x = batch_norm(name=name+"_bw")(x)
    else:
        raise Exception("need tensorflow addons")
        #x = tfa.layers.GroupNormalization(name=name+"_bw")(x)
    x = Activation("relu",name=name+"_activation")(x)
    return x

def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):
    x_deep= Conv2DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(x_deep)
    x_deep = batch_norm()(x_deep)   
    x_deep = LeakyReLU(alpha=0.1)(x_deep)
    x = Concatenate()([x_shallow, x_deep])
    x = Conv2D(out_ch, kernel_size=1, strides=1, padding="same")(x)
    x = batch_norm()(x)   
    x = LeakyReLU(alpha=0.1)(x)
    return x

def aggregation(skip_connections, output_layer_n, prefix=""):
    x_1= cbr(skip_connections["c1"], output_layer_n, 1, 1,prefix+"aggregation_1")
    x_1 = aggregation_block(x_1, skip_connections["c2"], output_layer_n, output_layer_n)
    x_2= cbr(skip_connections["c2"], output_layer_n, 1, 1,prefix+"aggregation_2")
    x_2 = aggregation_block(x_2, skip_connections["c3"], output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_3 = cbr(skip_connections["c3"], output_layer_n, 1, 1,prefix+"aggregation_3")
    x_3 = aggregation_block(x_3, skip_connections["c4"], output_layer_n, output_layer_n)
    x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_4 = cbr(skip_connections["c4"], output_layer_n, 1, 1,prefix+"aggregation_4")
    skip_connections_out=[x_1,x_2,x_3,x_4]
    return skip_connections_out

def effv2_encoder(inputs, is_train, from_scratch, model_name="s"):
    skip_connections={}
    pretrained_model = effnetv2_model.get_model('efficientnetv2-{}'.format(model_name), 
                                                model_config={"bn_type":"tpu_bn" if USE_TPU else None},
                                                include_top=False, 
                                                pretrained=False,
                                                training=is_train,
                                                input_shape=(None,None,3),
                                                input_tensor=inputs,
                                                with_endpoints=True)
    if not from_scratch:
        pretrained_model.load_weights(WEIGHT_DIR + 'effv2-{}-21k.h5'.format(model_name), by_name=True, skip_mismatch=True)    

    skip_connections["c1"] = pretrained_model.output[1]
    skip_connections["c2"] = pretrained_model.output[2]
    skip_connections["c3"] = pretrained_model.output[3]
    skip_connections["c4"] = pretrained_model.output[4]
    x = pretrained_model.output[5]

    return x, skip_connections


def decoder(inputs, skip_connections, use_batchnorm=True, 
            num_channels = 32, minimum_stride=2, max_stride=128,
            prefix=""):
    if not minimum_stride in [1,2,4,8]:
        raise Exception("minimum stride must be 1 or 2 or 4 or 8")
    if not max_stride in [32,64,128]:
        raise Exception("maximum stride must be 32 or 64 or 128")
    outs = []
    skip_connections = aggregation(skip_connections, num_channels, prefix=prefix)
    
    x = Dropout(0.2,noise_shape=(None, 1, 1, 1),name=prefix+'top_drop')(inputs)
    
    if max_stride>32:#more_deep        
        x_64 = cbr(x, 256, 3, 2,prefix+"top_64", use_batchnorm=use_batchnorm)
        if max_stride>64:
            x_128 = cbr(x_64, 256, 3, 2,prefix+"top_128", use_batchnorm=use_batchnorm)
            outs.append(x_128)
            x_64u = UpSampling2D(size=(2, 2))(x_128)
            x_64 = Concatenate()([x_64, x_64u])
        x_64 = cbr(x_64, 256, 3, 1,prefix+"top_64u", use_batchnorm=use_batchnorm)
        outs.append(x_64)
        x_32u = UpSampling2D(size=(2, 2))(x_64)
        x = Concatenate()([x, x_32u])    
    #x = Lambda(add_coords)(x)    
    x = cbr(x, num_channels*16, 3, 1,prefix+"decode_1", use_batchnorm=use_batchnorm)
    outs.append(x)
    x = UpSampling2D(size=(2, 2))(x)#8->16 tconvのがいいか

    x = Concatenate()([x, skip_connections[3]])
    x = cbr(x, num_channels*8, 3, 1,prefix+"decode_2", use_batchnorm=use_batchnorm)
    outs.append(x)
    x = UpSampling2D(size=(2, 2))(x)#16->32
    
    x = Concatenate()([x, skip_connections[2]])
    x = cbr(x, num_channels*4, 3, 1,prefix+"decode_3", use_batchnorm=use_batchnorm)
    outs.append(x)
   
    if minimum_stride<=4:
        x = UpSampling2D(size=(2, 2))(x)#32->64 
        x = Concatenate()([x, skip_connections[1]])
        x = cbr(x, num_channels*2, 3, 1,prefix+"decode_4", use_batchnorm=use_batchnorm)
        outs.append(x)
    if minimum_stride<=2:    
        x = UpSampling2D(size=(2, 2))(x)#64->128
        x = Concatenate()([x, skip_connections[0]])
        x = cbr(x, num_channels, 3, 1,prefix+"decode_5", use_batchnorm=use_batchnorm)
        outs.append(x)
    if minimum_stride==1:
        x = UpSampling2D(size=(2, 2))(x)#128->256
        outs.append(x)
    return outs


def FCOS_loss(multi_out_offset, 
              multi_out_center, 
              targets_box, 
              scale_min=4.0, minimum_stride=2, assign_by_box=True):
    #original implementation assigns ground truth by offset (not box size)
    #get ground truth matching the shape of prediction
    reg_target, centerness_target, positive_bool = Lambda(get_ground_truth_FCOS, arguments={"scale_min":scale_min, "assign_by_box":assign_by_box, "minimum_stride":minimum_stride})([targets_box, multi_out_offset])#, targets_class, multi_out_class])
    
    #concat multiscale prediction
    #multi_out_class = Lambda(flatten_and_concat_with_box)(multi_out_class)
    multi_out_offset = Lambda(flatten_and_concat_with_box)(multi_out_offset)
    multi_out_center = Lambda(flatten_and_concat_with_box)(multi_out_center)
        
    
    out_offsets = Lambda(box_iou_loss, name="out_offsets")([positive_bool, reg_target, multi_out_offset])
    out_centerness = Lambda(centerness_loss, arguments={"use_focal_loss":assign_by_box}, name="out_centerness")([centerness_target, multi_out_center])
    outputs = [out_offsets, out_centerness]
    loss = {"out_offsets": dummy_loss, "out_centerness":dummy_loss}
    loss_weights = {"out_offsets": 1.0, "out_centerness":1.0,}
    
    return outputs, loss, loss_weights

def dummy_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred) 

def center_exist_mask(func, threshold=1e-7):
    def inner(inputs):
        outputs = []
        #score_true = tf.squeeze(inputs[0])
        #print("AAAAAAA", tf.shape(inputs[0]))
        #centers = tf.reshape(inputs[0], [tf.shape(inputs[0])[0],tf.shape(inputs[0])[1],tf.shape(inputs[0])[2]])
        centers = tf.reshape(inputs[0], tf.shape(inputs[0])[:-1])
        centers = tf.reshape(inputs[0], tf.shape(inputs[0])[:-1])

        center_mask = centers>threshold
        center_weight = tf.reshape(tf.boolean_mask(centers, center_mask),[-1,1])#for mix up
        #print("S", score_true.shape)
        for input_array in inputs[1:]:
            #squeezeしようと思ったが現状不要だった…
            int_shape = input_array.get_shape().as_list()
            if int_shape[-1]==1:#squeezeでよかったな
                input_array = tf.reshape(input_array, tf.shape(input_array)[:-1])
            if len(int_shape)>3:
                start_from=3
            else:
                start_from=2
            output_shape = tf.concat(([-1],tf.shape(input_array)[start_from:]),axis=0)#[-1]+list(tf.shape(array)[3:])
            #output_shape = [-1]+list(tf.shape(input_array)[3:])
            output_array = tf.boolean_mask(input_array, center_mask)
            output_array = tf.reshape(output_array, output_shape)
            #print(output_array.shape)
            outputs.append(output_array)
        return func(outputs+[center_weight])
    return inner


def ciou_loss(gt_top, gt_left, gt_bottom, gt_right,
              pred_top, pred_left, pred_bottom, pred_right):
    """
    all inputs is the offset from center of the pixel. positive values.
    """
    enclosing_top = tf.maximum(gt_top, pred_top)
    enclosing_left = tf.maximum(gt_left, pred_left)
    enclosing_bottom = tf.maximum(gt_bottom, pred_bottom)
    enclosing_right = tf.maximum(gt_right, pred_right)
    intersection_top = tf.minimum(gt_top, pred_top)
    intersection_left =  tf.minimum(gt_left, pred_left)
    intersection_bottom = tf.minimum(gt_bottom, pred_bottom)
    intersection_right = tf.minimum(gt_right, pred_right)
    
    gt_width = gt_left + gt_right
    gt_height = gt_top + gt_bottom
    pred_width = pred_left + pred_right
    pred_height = pred_top + pred_bottom

    # distance to calculate DIoU.   not plus, be careful
    box_dist = (((gt_top-gt_bottom)-(pred_top-pred_bottom))/2)**2 + (((gt_left-gt_right)-(pred_left-pred_right))/2)**2
    diagonal_dist = (enclosing_top + enclosing_bottom)**2 + (enclosing_left + enclosing_right)**2
    
    gt_area = (gt_bottom + gt_top) * (gt_right + gt_left)
    pred_area = (pred_bottom + pred_top) * (pred_right + pred_left)
    intersection_area = (intersection_bottom + intersection_top) * (intersection_right + intersection_left)
    union_area = gt_area + pred_area - intersection_area
    iou = intersection_area / (union_area + 1e-7)#tf.math.divide_no_nan(intersection_area, union_area)
    
    #enclosing_area = (enclosing_bottom + enclosing_top) * (enclosing_right + enclosing_left)
    #giou = iou - (enclosing_area - union_area)/enclosing_area
    #giou_loss = 1.0 - giou
    
    diou = iou - box_dist / (diagonal_dist + 1e-7)#tf.math.divide_no_nan)
    # completed iou considers aspect ratio
    gt_aspect_ratio = gt_width / (gt_height + 1e-7)
    pred_aspect_ratio = pred_width / (pred_height + 1e-7)
    v = ((tf.math.atan(gt_aspect_ratio)
          - tf.math.atan(pred_aspect_ratio)) * 2 / tf.constant(np.pi, tf.float32)) ** 2
    alpha = tf.math.divide_no_nan(v, 1 - iou + v)

    ciou = diou - alpha * v
    ciou_loss = 1 - ciou  
    
    """
    iou = tf.math.divide_no_nan(intersection_area, union_area)
    
    #enclosing_area = (enclosing_bottom + enclosing_top) * (enclosing_right + enclosing_left)
    #giou = iou - (enclosing_area - union_area)/enclosing_area
    #giou_loss = 1.0 - giou
    
    diou = iou - tf.math.divide_no_nan(box_dist, diagonal_dist)
    # completed iou considers aspect ratio
    v = ((tf.math.atan(tf.math.divide_no_nan(gt_width, gt_height))
          - tf.math.atan(tf.math.divide_no_nan(pred_width, pred_height))) * 2 / tf.constant(np.pi, tf.float32)) ** 2
    alpha = tf.math.divide_no_nan(v, 1 - iou + v)

    ciou = diou - alpha * v
    ciou_loss = 1 - ciou    
    """
    return ciou_loss

@center_exist_mask
def bce_loss(inputs):
    y_true, y_pred, c_weight = inputs
    print(y_true, y_pred, c_weight)
    c_weight = tf.reshape(c_weight, [-1,1])
    y_true = tf.cast(y_true, y_pred.dtype)
    epsilon = K.epsilon()    
    y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    loss = - y_true * tf.math.log(y_pred) - (1.0-y_true) * tf.math.log(1.0-y_pred)
    return tf.reduce_sum(loss*c_weight)/(tf.reduce_sum(c_weight)+1e-7)
    
def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    """binary(?) focal loss 
    TODO add loss weight"""
    y_true = tf.cast(y_true, y_pred.dtype)
    num_object = tf.reduce_sum(y_true)#tf.where(y_true==1, 1.0, 0.0))
    #pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    #pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    # for soft target
    epsilon = K.epsilon()
    
    pt_1 = tf.where(y_true>epsilon, y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(y_true<=epsilon, y_pred, tf.zeros_like(y_pred))
    #pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    
    pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
    pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)
    loss = - (alpha * ((1.-pt_1)**gamma) * tf.math.log(pt_1)) *  y_true\
        - ((1-alpha) * (pt_0**gamma) * tf.math.log(1.-pt_0)) * (1.-y_true)
    return loss, num_object

def focal_loss_with_weight(y_true, y_pred, gamma=2., alpha=.25, threshold = 0.1):
    """focal loss 
    TODO add loss weight"""
    
    y_true = tf.cast(y_true, y_pred.dtype)
    num_object = tf.reduce_sum(y_true)#tf.where(y_true==1, 1.0, 0.0))
    pt_1 = tf.where(y_true>=threshold, y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(y_true<threshold, y_pred, tf.zeros_like(y_pred))
    weights = tf.where(y_true>threshold, y_true, 0.1)#positive is very rare.
    epsilon = K.epsilon()
    pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
    pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)
    loss = - (alpha * ((1.-pt_1)**gamma) * tf.math.log(pt_1)) - ((1-alpha) * (pt_0**gamma) * tf.math.log(1.-pt_0))        
    return loss*weights, num_object

def center_loss(inputs):
    y_true, y_pred = inputs
    loss, num_object = focal_loss(y_true, y_pred)
    loss = tf.reduce_sum(loss) / (num_object + 1)
    return loss

def centerness_loss(inputs, use_focal_loss=False, gamma=2., alpha=.25):
    y_true, y_pred = inputs
    y_true = tf.cast(y_true, y_pred.dtype)
    epsilon = K.epsilon()
    if use_focal_loss:
        positive_samples = (y_true>=1.0-epsilon)
        num_box = tf.reduce_sum(tf.cast(positive_samples, tf.float32))
        pt_1 = tf.where(positive_samples, y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(y_true<1.0-epsilon, y_pred, tf.zeros_like(y_pred))
        weights = y_true
        pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
        pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)
        #small penalty for wrong prediction around true center 
        loss = - (alpha * ((1.-pt_1)**gamma) * tf.math.log(pt_1)) - (1.-weights)*((1-alpha) * (pt_0**gamma) * tf.math.log(1.-pt_0))        
        loss = tf.reduce_sum(loss)/(num_box+1.0)
    else:#BCE (logloss)
        y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        loss = - y_true * tf.math.log(y_pred) - (1.0-y_true) * tf.math.log(1.0-y_pred)
        loss= tf.reduce_mean(loss)
    return loss

@center_exist_mask
def box_iou_loss(inputs):
    #gt_top, gt_left, gt_bottom, gt_right, pred_top, pred_left, pred_bottom, pred_right
    gt_offsets, pred_offsets, c_weight = inputs
    c_weight = tf.reshape(c_weight, [-1])
    loss = ciou_loss(gt_offsets[...,0], gt_offsets[...,1], gt_offsets[...,2], gt_offsets[...,3],
                     pred_offsets[...,0], pred_offsets[...,1], pred_offsets[...,2], pred_offsets[...,3])
    loss = tf.reduce_sum(loss*c_weight)/(tf.reduce_sum(c_weight)+1e-7)
    return loss

@center_exist_mask
def mse_loss(inputs):
    gt_vec, pred_vec = inputs
    loss = tf.reduce_mean((gt_vec- pred_vec)**2)
    return loss

def get_ground_truth_FCOS(inputs, scale_min=4.0, minimum_stride=2, 
                          center_sampling_threshold=0.5, assign_by_box=False):
    #targets_box, targets_class, ref_multiscale_img = inputs
    targets_box, ref_multiscale_img = inputs
    strides_list = []
    height_width_list = []
    reg_min_max_multiscale = []
    gt_box_offsets_multiscale = []
    #gt_box_class_multiscale = []
    _, base_height, base_width, _ = tf.unstack(tf.shape(ref_multiscale_img[0]))
    base_height = minimum_stride*base_height# shape of input scale
    base_width = minimum_stride*base_width#
    for i, img in enumerate(ref_multiscale_img):
        b, h, w, ch = tf.unstack(tf.shape(img))
        strides_list.append(2**i)
        height_width_list.append([h,w])
        gt_box_offsets= get_box_offsetimg_from_gt(targets_box, 
                                                  #targets_class, 
                                                  base_height, base_width, height_width=[h,w])
        if len(ref_multiscale_img)==1:
            reg_min_max = [[[[0.0, np.inf]]]]
        elif i==0:
            reg_min_max = [[[[0.0, np.inf]]]]
        elif i== len(ref_multiscale_img)-1:
            reg_min_max = [[[[scale_min*(2**(i-1)), np.inf]]]]
        else:
            reg_min_max = [[[[scale_min*(2**(i-1)), scale_min*(2**i)]]]]
        reg_min_max_multiscale.append(tf.tile(tf.constant(reg_min_max),[1,1,h*w,1]))
        gt_box_offsets_multiscale.append(gt_box_offsets)
        #gt_box_class_multiscale.append(gt_box_class)
    #concat multiscale axis2 is sum of Hi*Wi
    gt_box_offsets_multiscale = tf.concat(gt_box_offsets_multiscale, axis=2)
    #gt_box_class_multiscale = tf.concat(gt_box_class_multiscale, axis=2)
    inside_box = tf.reduce_min(gt_box_offsets_multiscale, axis=-1, keepdims=True)>1e-7
    #valid_box = tf.reduce_max(gt_box_class_multiscale, axis=-1, keepdims=True)>1e-7
    gt_box_offsets_multiscale = gt_box_offsets_multiscale * tf.cast(inside_box, tf.float32)# * tf.cast(valid_box, tf.float32)
    #gt_box_class_multiscale = gt_box_class_multiscale * tf.cast(inside_box, tf.float32) * tf.cast(valid_box, tf.float32)
    reg_min_max_multiscale = tf.concat(reg_min_max_multiscale, axis=2)
    scale_assign_bool = scale_assignment(gt_box_offsets_multiscale, reg_min_max_multiscale, by_box=assign_by_box)
    centerness_target = gt_centerness(gt_box_offsets_multiscale, by_box=assign_by_box)
    center_sampling_bool = centerness_target > center_sampling_threshold
    positive_bool = tf.math.logical_and(scale_assign_bool, center_sampling_bool)
    positive_bool = tf.math.logical_and(positive_bool, inside_box)
    #positive_bool = tf.math.logical_and(positive_bool, valid_box)
    centerness_target = centerness_target * tf.cast(scale_assign_bool, tf.float32)
    #class_target = gt_box_class_multiscale * tf.cast(positive_bool, tf.float32)
    reg_target = gt_box_offsets_multiscale
    reg_target = tf.where(reg_target==0, tf.constant(np.inf), reg_target)
    # take max or minimum at axis of num_box
    centerness_target = tf.reduce_max(centerness_target, axis=1)
    #class_target = tf.reduce_max(class_target, axis=1)
    reg_target = tf.reduce_min(reg_target, axis=1)
    positive_bool = tf.reduce_max(tf.cast(positive_bool, tf.float32), axis=1)
    #return class_target, reg_target, centerness_target, positive_bool
    return reg_target, centerness_target, positive_bool


def gt_centerness(reg_targets, by_box=False):
    left_right = reg_targets[..., 1::2]
    top_bottom = reg_targets[..., 0::2]   
    centerness = (tf.reduce_min(left_right, axis=-1, keepdims=True) / (tf.reduce_max(left_right, axis=-1, keepdims=True)+1e-7)) * (tf.reduce_min(top_bottom, axis=-1, keepdims=True) / (tf.reduce_max(top_bottom, axis=-1, keepdims=True)+1e-7))
    centerness= tf.math.sqrt(centerness)
    if by_box:
        centerness = tf.where(tf.logical_and(centerness==tf.reduce_max(centerness, axis=2, keepdims=True), centerness>1e-7), 1.0, centerness)
    return centerness

def scale_assignment(gt_box_offsets, reg_min_max, by_box=False):
    """
    gt_box_offsets : [Batch,Num_box,sum of H*W, 4]
    reg_min_max : [1, 1, sum of H*W, 2]
    returns:
        boolean_mask [Batch,Num_box,sum of H*W, 1]
    """
    if by_box:
        ref_offset = 2.0 * tf.reduce_mean(gt_box_offsets, axis=-1, keepdims=True)
    else:#original implementation
        ref_offset = tf.reduce_max(gt_box_offsets, axis=-1, keepdims=True)
    return tf.math.logical_and(ref_offset>=reg_min_max[...,0:1], ref_offset<reg_min_max[...,1:2])


def get_box_offsetimg_from_gt(box_gt, 
                              #box_class, 
                              base_height, base_width, height_width):
    """
    inputs:
        box_gt [batch, num_gt, 4(top,left,bottom,right)]
        box_class [batch, num_gt, 4(top,left,bottom,right)]
        resolutions_list is list of [height, width] for each scale
    returns:
        box_offsets [batch, height*width, 4(positive offsets to 4 directions)]
        box_class [batch, height*width, 2(num_class)]
    """
    out_height, out_width = height_width
    batch, num_box, _ = tf.unstack(tf.shape(box_gt))
    box_gt = box_gt[:,:,tf.newaxis,tf.newaxis,:]#[batch, num_gt, 1, 1, 4(top,left,bottom,right)]
    #[batch=1, h, w, 5], score, top_offset, left_ofst, bottom_ofst, right_ofst
    #out_height, out_width = box_img_shape[1], box_img_shape[2]
    base_height = tf.cast(base_height,tf.float32)
    base_width = tf.cast(base_width,tf.float32)
    s = 0.5 * (base_height/tf.cast(out_height,tf.float32))
    e = base_height - s
    top_offset = -box_gt[...,0:1] + tf.reshape(tf.linspace(s, e, out_height),[1,1,-1,1,1])
    bottom_offset = box_gt[...,2:3] - tf.reshape(tf.linspace(s, e, out_height),[1,1,-1,1,1])
    top_offset = tf.tile(top_offset, [1,1,1,out_width,1])
    bottom_offset = tf.tile(bottom_offset, [1,1,1,out_width,1])
    s = 0.5 * (base_width/tf.cast(out_width,tf.float32))
    e = base_width - s
    left_offset = -box_gt[...,1:2] + tf.reshape(tf.linspace(s, e, out_width),[1,1,1,-1,1])
    right_offset = box_gt[...,3:4] - tf.reshape(tf.linspace(s, e, out_width),[1,1,1,-1,1])
    left_offset = tf.tile(left_offset, [1,1,out_height,1,1])
    right_offset = tf.tile(right_offset, [1,1,out_height,1,1])
    box_offsets = tf.concat([top_offset,left_offset,bottom_offset,right_offset], axis=-1)
    box_offsets = tf.reshape(box_offsets, [batch, num_box, -1, 4])
    #box_class = tf.tile(box_class[:,:,tf.newaxis,:], [1,1,out_height*out_width,1])
    return box_offsets#, box_class
  
    
def flatten_and_concat_with_box(multiscale_inputs):#, size_min=None, size_max=None):
    """
    reshape each-scale output [batch, height, width, channnel] to [batch, height*width, channel]
    """
    multiscale_outputs = []
    for i, inputs in enumerate(multiscale_inputs):
        batch = tf.unstack(tf.shape(inputs))[0]
        channels = tf.unstack(tf.shape(inputs))[-1]
        outputs = tf.reshape(inputs, [batch, -1, channels])
        multiscale_outputs.append(outputs)
    multiscale_outputs = tf.concat(multiscale_outputs, axis=1)
    return multiscale_outputs

def detection_head(ch_num):
    layers = []
    for i in range(1,4):
        name = "reg_branch_{}".format(i)
        block = [Conv2D(ch_num, kernel_size=3, strides=1, use_bias=False, padding="same", name=name+"_conv"),
                 BatchNormalization(name=name+"_bw"),
                 Activation("relu", name=name+"_activation")]
        layers = layers + block
    out_layer_reg = Conv2D(4, kernel_size=3, strides=1, activation="sigmoid", padding="same", name="fcos_reg_out")
    out_layer_center = Conv2D(1, kernel_size=3, strides=1, activation="sigmoid", padding="same", 
                        bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)), name="fcos_center_out")
    reg_branch = tf.keras.Sequential(layers+[out_layer_reg], name="fcos_reg_branch")
    reg_branch_center = tf.keras.Sequential(layers+[out_layer_center], name="fcos_reg_branch_center")
    return reg_branch, reg_branch_center

def get_box_tlbr_from_offsetimg(box_img, base_height, base_width, hw_rate=1):
    """
    inputs:
        box_img [batch, height, width, 4(positive offsets to 4 directions)]
    returns:
        box_coords [batch, height*width(==num_box), 4]
    """
    base_height = tf.cast(base_height, tf.float32)
    base_width = tf.cast(base_width, tf.float32)
    box_img_shape = tf.shape(box_img)#[batch=1, h, w, 5], score, top_offset, left_ofst, bottom_ofst, right_ofst
    out_height, out_width = box_img_shape[1], box_img_shape[2]
    s_h = 0.5 * (base_height/tf.cast(out_height,tf.float32))
    e_h = base_height - s_h
    s_w = 0.5 * (base_width/tf.cast(out_width,tf.float32))
    e_w = base_width - s_w
    top = -box_img[...,0:1]*hw_rate + tf.reshape(tf.linspace(s_h, e_h, out_height),[1,-1,1,1])
    left = -box_img[...,1:2]*hw_rate + tf.reshape(tf.linspace(s_w, e_w, out_width),[1,1,-1,1])
    bottom = box_img[...,2:3]*hw_rate + tf.reshape(tf.linspace(s_h, e_h, out_height),[1,-1,1,1])
    right = box_img[...,3:4]*hw_rate + tf.reshape(tf.linspace(s_w, e_w, out_width),[1,1,-1,1])
    box_coords = tf.concat([top,left,bottom,right], axis=-1)
    #box_coords = tf.reshape(box_coords, [box_img_shape[0], -1, 4])
    return box_coords

def soft_nms_layer(inputs, score_threshold=0.10, max_output_size=30, iou_thresh=0.70, sigma=0.5):
    boxes, scores = inputs
    boxes = boxes[0,...]
    scores = scores[0,...,0]#*0.21
    
    #score_threshold = 0.10
    
    ##先にbooleanmaskとってから流すほうが早い？
    score_mask = scores > score_threshold
    boxes = tf.boolean_mask(boxes, score_mask, axis=0)
    scores = tf.boolean_mask(scores, score_mask, axis=0)
    
    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                                            boxes, scores, 
                                            max_output_size, 
                                            iou_threshold=iou_thresh, 
                                            score_threshold=score_threshold,
                                            soft_nms_sigma=sigma,#0.5,
                                            )
    selected_boxes = tf.gather(boxes, selected_indices)
    return selected_boxes, selected_scores
    
    
def build_detection_model(input_shape=(256,256,3),
             backbone="effv2s", minimum_stride=4, max_stride = 64,
             is_train=True,
             from_scratch=False,
             num_boxes = None,
             return_heatmap=False,
             include_nms=True):
    input_rgb = Input(input_shape, name="input_rgb")#256,256,3
    enc_in = input_rgb
    model_names = {"effv2s":"s", "effv2m":"m", "effv2l":"l", "effv2xl":"xl"}
    if backbone not in model_names.keys():
        raise Exception("check backbone name")
    x, skip_connections = effv2_encoder(enc_in, is_train, from_scratch, model_name = model_names[backbone])
            
    outs = decoder(x, skip_connections, use_batchnorm=True, 
                   num_channels=32, max_stride=max_stride, minimum_stride=minimum_stride)
    
    ch_num = 64
    reg_branch, center_branch = detection_head(ch_num)
    
    x = outs[-1]#only highest resolution
    box_offsets = reg_branch(x)
    box_centers = center_branch(x)

    scale_rate = 50.
    box_offsets = Lambda(lambda x: x*tf.constant(scale_rate,tf.float32), name="box_offsets_scale_adjust")(box_offsets)
    
    multi_out_offset = [box_offsets]
    multi_out_center = [box_centers]
    
    if is_train:
        inputs_box = Input([num_boxes, 4], name="inputs_box_tlbr_input_scale")
        detection_inputs = [inputs_box]
        detection_outputs, detection_loss, detection_loss_weights = FCOS_loss(multi_out_offset, multi_out_center, inputs_box, scale_min=0.0, minimum_stride=minimum_stride)
        
    else:
        if return_heatmap:
            detection_outputs = [box_offsets, box_centers]
        else:
            multi_out_offset = Lambda(lambda x: [get_box_tlbr_from_offsetimg(box_img, minimum_stride*tf.shape(x[0])[1], minimum_stride*tf.shape(x[0])[2]) for box_img in x])(multi_out_offset)
            boxes = Lambda(flatten_and_concat_with_box)(multi_out_offset)
            scores = Lambda(flatten_and_concat_with_box)(multi_out_center)
            if include_nms:
                boxes, scores = Lambda(soft_nms_layer)([boxes, scores])
            detection_outputs = [boxes, scores]
        detection_inputs = []
        
        detection_loss = {}
        detection_loss_weights = {}
        
    model = Model([input_rgb] + detection_inputs, detection_outputs)
    return model, detection_loss, detection_loss_weights    
    
if __name__ == "__main__":
    m,_,_ = build_detection_model(is_train=False)
    print(m.summary())