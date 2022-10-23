# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 16:01:00 2021

@author: kmat
"""
import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def transform_points(transmatrix, points):
    x, y = tf.split(points, 2, axis=-1)
    xyones = tf.concat([x,y,tf.ones_like(x)], axis=-1)
    trans_points = tf.matmul(xyones, transmatrix, transpose_b=True)[...,:2]
    return trans_points

def get_nearest_distance(sources, targets):
    dist_matrix = tf.reduce_sum((sources[:,tf.newaxis,:] - targets[tf.newaxis,:,:])**2, axis=-1)
    dist_to_near_target = tf.reduce_min(dist_matrix, axis=1)
    dist_to_near_source = tf.reduce_min(dist_matrix, axis=0)
    return dist_to_near_target, dist_to_near_source

def search_nearest_error(sources, targets):
    #[batch, num_source, num_target]
    dist_matrix = tf.reduce_sum((sources[tf.newaxis,:,tf.newaxis,:] - targets[tf.newaxis,tf.newaxis,:,:])**2, axis=-1)
    assignments = tf_linear_sum_assignment_batch(dist_matrix)[0]
    assigned_targets = tf.gather(targets, assignments)

    target_idx = tf.cast(tf.range(tf.shape(targets)[0])[:,tf.newaxis], tf.float32)
    assigned_index = tf.cast(assignments, tf.float32)[tf.newaxis,:]
    not_assigned_idx_target = tf.boolean_mask(target_idx[:,0], 
                                              tf.reshape(tf.reduce_all(target_idx != assigned_index, axis=-1), [tf.shape(targets)[0],]))
    not_assigned_targets = tf.gather(targets, tf.cast(not_assigned_idx_target, tf.int32))
    error = tf.reduce_mean(tf.reduce_sum((sources - assigned_targets)**2, axis=-1))
    return error, assigned_targets, assignments, not_assigned_targets

def random_circle_search(targets, sources, num_try):
    # randomly search the area whose number of targets are same as the source
    num_points, _ = tf.unstack(tf.shape(sources))

    base_radius = tf.math.sqrt(tf.reduce_max(tf.reduce_sum(sources[...,:2]**2, axis=-1)))
    targets_xy_min = tf.reduce_min(targets[...,:2], axis=0)
    targets_xy_max = tf.reduce_max(targets[...,:2], axis=0)
    
    centers = tf.random.uniform((num_try, 2), minval=targets_xy_min, maxval=targets_xy_max)
    radius = base_radius * tf.math.exp(tf.random.uniform((num_try,), -0.25, 0.25))
    
    dists_from_centers = tf.math.sqrt(tf.reduce_sum((targets[tf.newaxis,:,:] - centers[:,tf.newaxis,:])**2, axis=-1))
    num_inside_circle = tf.reduce_sum(tf.cast(dists_from_centers < radius[:,tf.newaxis], tf.int32), axis=1)
    ok_mask = tf.math.logical_and(num_points-2<=num_inside_circle, num_inside_circle<=num_points+2)
    centers = tf.boolean_mask(centers, ok_mask)
    return centers


def points2points_fitting(targets, sources, num_iter=6, l2_reg=0.1, rot_init=0.):
    
    def get_transmatrix(k, rz, tx, ty):
        """
        k : zoom ratio.
        rz : rotation.
        tx : x offset.
        ty : z offset
        shape [batch]
        
        returns:
            transmatrix with shape [batch, 3, 3]
        """
        exp_k = tf.math.exp(k)
        sin = tf.math.sin(rz)
        cos = tf.math.cos(rz)
        mat = tf.stack([[exp_k*cos, -exp_k*sin, exp_k*tx],
                        [exp_k*sin, exp_k*cos, exp_k*ty],
                        [tf.zeros_like(k), tf.zeros_like(k), tf.ones_like(k)]])
        mat = tf.transpose(mat, [2,0,1])
        return mat
        
    def transform_points(transmatrix, points):
        x, y = tf.split(points, 2, axis=-1)
        xyones = tf.concat([x,y,tf.ones_like(x)], axis=-1)
        trans_points = tf.matmul(xyones, transmatrix, transpose_b=True)[...,:2]
        return trans_points
    
    def get_derivative_at(k, rz, tx, ty, points):
        dev = 1e-5
        original = transform_points(get_transmatrix(k, rz, tx, ty), points)
        dxy_dk = (transform_points(get_transmatrix(k+dev, rz, tx, ty), points) - original) / dev
        dxy_drz = (transform_points(get_transmatrix(k, rz+dev, tx, ty), points) - original) / dev
        dxy_dtx = (transform_points(get_transmatrix(k, rz, tx+dev, ty), points) - original) / dev
        dxy_dty = (transform_points(get_transmatrix(k, rz, tx, ty+dev), points) - original) / dev
        return original, dxy_dk, dxy_drz, dxy_dtx, dxy_dty
    
    def normal_equation(X, Y, gn_rate=0.9, gd_rate=0.1):
        #not in use
        #Gaussian Neuton combind with Gradient descent
        XtX = tf.matmul(X, X, transpose_a=True)
        #XtX_inv = np.linalg.inv(XtX)
        #XtX_inv = 1.0*XtX_inv + 0.5*np.eye(len(XtX))
        #results = np.dot(XtX_inv, np.dot(X.T, Y))
        
        results_neuton = tf.linalg.solve(XtX, tf.matmul(X, Y, transpose_a=True))
        results_gaussian = tf.matmul(tf.tile(tf.eye(XtX.shape[1])[tf.newaxis,...], [XtX.shape[0],1,1]), tf.matmul(X, Y, transpose_a=True))
        results = gn_rate*results_neuton + gd_rate*results_gaussian
        
        return results
    
    # initial_values
    batch, num_points = tf.unstack(tf.shape(targets))[:2]
    k = 0.0 * tf.ones((batch), tf.float32)#expで取る。ネガティブでんし。
    rz = rot_init * tf.ones((batch), tf.float32)
    tx = 0.0 * tf.ones((batch), tf.float32)
    ty = 0.0 * tf.ones((batch), tf.float32)
    
    source_origin = sources
    for i in range(num_iter):
        currents, dxy_dk, dxy_rz, dxy_dtx, dxy_dty = get_derivative_at(k, rz, tx, ty, source_origin)
        b = tf.reshape(targets-currents, [batch, num_points*2, 1])#xy flatten
        a = tf.stack([dxy_dk, dxy_rz, dxy_dtx, dxy_dty], axis=-1)
        a = tf.reshape(a, [batch, num_points*2, 4])
        updates = tf.linalg.lstsq(a, b, l2_regularizer=l2_reg, fast=True)#batch, 4, 1
        
        k = k + updates[:,0,0]
        rz = rz + updates[:,1,0]
        tx = tx + updates[:,2,0]
        ty = ty + updates[:,3,0]
    trans_matrix = get_transmatrix(k, rz, tx, ty)
    trans_sources = transform_points(trans_matrix, sources)
    return trans_sources, trans_matrix, k, rz, tx, ty


@tf.function(input_signature=(tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.int32),
                              tf.TensorSpec(shape=[], dtype=tf.bool),
                              tf.TensorSpec(shape=[], dtype=tf.bool),
                              tf.TensorSpec(shape=[], dtype=tf.bool),
                              tf.TensorSpec(shape=[], dtype=tf.bool),
                              tf.TensorSpec(shape=[3], dtype=tf.float32),
                              tf.TensorSpec(shape=[3], dtype=tf.float32),
                              tf.TensorSpec(shape=[2,3], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.float32),##増やした
                              )
             )
def make_icp_inputs(targets, 
                       sources,
                       confidence,
                       targets_team,
                       sources_team, 
                       num_trial,
                       is_sideline=True,#Sideline or Endzone
                       team_provided=False,
                       use_provided_params = False,
                       use_random_params = True,
                       zoom_params = tf.zeros((3), tf.float32),
                       rz_params = tf.zeros((3), tf.float32),#[0.0, 0.5, 0.01],#mean, std, l2 penalty
                       txy_params = tf.zeros((2,3), tf.float32),
                       confidence_threshold = 0.0,
                       team_weight = 0.1):
    centering_by_circle = True
    
    targets_team = targets_team * team_weight
    sources_team = sources_team * team_weight
    
    valid_mask = confidence > confidence_threshold
    all_invalid_mask = confidence > 1.0
    
    #center調整
    s_mean = tf.reduce_mean(sources, axis=0, keepdims=True)
    sources = sources - s_mean
    
    
    if is_sideline:
        mean_k = 0.0
        sigma_k = 0.5
        min_rz_0 = -0.6
        max_rz_0 = 0.6
        min_rz_1 = np.pi - 0.6
        max_rz_1 = np.pi + 0.6
        txty_std_t = tf.math.reduce_std(targets, axis=0)
        txty_std_s = tf.math.reduce_std(sources, axis=0)
        dev_std = tf.math.abs(txty_std_t - txty_std_s)
        std_tx = dev_std[0] + 0.05
        std_ty = dev_std[1] + 0.05
        mean_k = (tf.math.log(tf.reduce_sum(txty_std_t**2) / tf.reduce_sum(txty_std_s**2)))/2.0
        
    else:
        mean_k = 0.0
        sigma_k = 0.5
        min_rz_0 = -0.6 + np.pi/2
        max_rz_0 = 0.6 + np.pi/2
        min_rz_1 = -0.6 - np.pi/2
        max_rz_1 = 0.6 - np.pi/2
        txty_std_t = tf.math.reduce_std(targets, axis=0)
        txty_std_s = tf.math.reduce_std(sources, axis=0)
        dev_std = tf.math.abs(txty_std_t - txty_std_s)
        std_tx = dev_std[0] + 0.05
        std_ty = dev_std[1] + 0.05
        mean_k = (tf.math.log(tf.reduce_sum(txty_std_t**2) / tf.reduce_sum(txty_std_s**2)))/2.0
 
    rz_mean = rz_params[0]
    rz_std = rz_params[1]
    rz_penalty = rz_params[2]
    
    if use_provided_params:
        #rz = tf.random.normal([num_trial], rz_mean, rz_std, tf.float32)
        rz = tf.random.uniform([num_trial], rz_mean-rz_std, rz_mean+rz_std, tf.float32)
        rz_penalty = rz_penalty
        #provided_xy = tf.reduce_mean(targets, axis=0, keepdims=True)

        #"""
        mean_tx = txy_params[0,0]
        mean_ty = txy_params[1,0]
        std_tx = txy_params[0,1]
        std_ty = txy_params[1,1]
        tx_penalty = txy_params[0,2]
        ty_penalty = txy_params[1,2]
        
        mean_k = zoom_params[0]
        sigma_k = zoom_params[1]
        k_penalty = zoom_params[2]
        
    else:
        rz_0 = tf.random.uniform([num_trial//2], min_rz_0, max_rz_0, tf.float32)        
        rz_1 = tf.random.uniform([num_trial-num_trial//2], min_rz_1, max_rz_1, tf.float32)
        rz = tf.concat([rz_0, rz_1], axis=0)
        rz_penalty = 1.0


        t_mean = tf.reduce_mean(targets, axis=0, keepdims=True)
        mean_tx = t_mean[0,0]
        mean_ty = t_mean[0,1]
        tx_penalty = 1.0
        ty_penalty = 1.0
        k_penalty = 1.0
        
    k = tf.random.normal([num_trial], mean_k, sigma_k, tf.float32)
    tx = tf.random.normal([num_trial], mean_tx, std_tx, tf.float32)
    ty = tf.random.normal([num_trial], mean_ty, std_ty, tf.float32)        
    
    if centering_by_circle:
        if use_random_params:
            txty_circle = random_circle_search(targets, sources, num_trial)
            num_points_circle, _ = tf.unstack(tf.shape(txty_circle))
            #tx = tf.concat([txty_circle[:,0], tx[:(num_trial-num_points_circle)]], axis=0)
            #ty = tf.concat([txty_circle[:,1], ty[:(num_trial-num_points_circle)]], axis=0)
            # half at maximum
            num_points_circle = tf.minimum(num_points_circle, num_trial//2)
            tx = tf.concat([txty_circle[:num_points_circle,0], tx[:(num_trial-num_points_circle)]], axis=0)
            ty = tf.concat([txty_circle[:num_points_circle,1], ty[:(num_trial-num_points_circle)]], axis=0)
    #tx = tf.random.uniform([], min_tx, max_tx, tf.float32)
    #ty = tf.random.uniform([], min_ty, max_ty, tf.float32)

    # 中央サンプル多め。OK？面積均一の方がいいかな？
    #radius = tf.random.uniform([], 0., farthest_dist, tf.float32)
    #angle = tf.random.uniform([], -np.pi, np.pi, tf.float32)
    #tx = radius * tf.math.sin(angle)
    #ty = radius * tf.math.cos(angle)
    targets = tf.concat([targets, tf.reshape(targets_team, [-1,1])], axis=-1)
    targets = tf.tile(targets[tf.newaxis,...], [num_trial, 1, 1])
    
    #concat team label (third axis)
    #現状、ソースは非固定ランダム。
    sources_team = tf.reshape(sources_team, [1,-1,1])
    
    if not team_provided:
        team_pos_neg = tf.cast(tf.random.uniform([num_trial,1,1], 0., 1., tf.float32)>0.5, tf.float32)
        sources_team = sources_team * team_pos_neg + (team_weight-sources_team) * (1.-team_pos_neg)
    else:
        sources_team = tf.tile(sources_team, [num_trial, 1, 1])
    
    sources = tf.tile(sources[tf.newaxis,...], [num_trial, 1, 1])
    #sources_team = tf.ones_like(sources)[...,:1]
    sources = tf.concat([sources, sources_team], axis=-1)
    #sources = tf.reshape(sources, [num_trial,-1,3])
    
    ##sources = tf.concat([sources, tf.reshape(sources_team, [-1,1])], axis=-1)
    ##sources = tf.tile(sources[tf.newaxis,...], [num_trial, 1, 1])

    l2_penalty = tf.stack([[k_penalty, rz_penalty, tx_penalty, ty_penalty]])
    l2_penalty = tf.tile(l2_penalty, [num_trial, 1])[...,tf.newaxis]
    
    #current_sources, current_matrix, current_redisual, current_assignment, current_raw_results = icp_fitting_2d_batch(targets, sources, valid_mask, num_iter=num_fitting_iter, batch_size=num_trial,
    #                                                                      k = k, rz = rz, tx = tx, ty = ty, 
    #         
    #"""
    return targets, sources, valid_mask, all_invalid_mask, k, rz, tx, ty, l2_penalty

@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                              
                              tf.TensorSpec(shape=[None], dtype=tf.bool),
                              tf.TensorSpec(shape=[None], dtype=tf.bool),
                              
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),

                              tf.TensorSpec(shape=[None, 4, 1], dtype=tf.float32),

                              tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              
                              tf.TensorSpec(shape=[], dtype=tf.int32),
                              tf.TensorSpec(shape=[], dtype=tf.int32),
                              tf.TensorSpec(shape=[], dtype=tf.int32),
                              )
             )
def random_icp_fitting(targets, 
                       sources,
                       valid_mask, all_invalid_mask,
                       k, rz, tx, ty,
                       l2_penalty,
                       st_cost_matrix, 
                       num_fitting_iter=20,
                       num_harddrop = 0,
                       num_softdrop=0,
                       ):
    try_drop = True
    
    
    
    current_sources, current_matrix, current_redisual, current_assignment, current_raw_results, current_source_selection, current_residual_xy = icp_fitting_2d_batch_confdrop_cost(targets, sources, valid_mask, cost=st_cost_matrix, num_iter=num_fitting_iter,
                                                                          k = k, rz = rz, tx = tx, ty = ty, 
                                                                          l2_penalty=l2_penalty,
                                                                          num_harddrop=num_harddrop, num_softdrop=num_softdrop)
    # length of assignment is shorter than sources, due to harddrop
    best_idx = tf.argmin(current_redisual)
    current_sources = tf.gather(current_sources, best_idx)
    current_matrix = tf.gather(current_matrix, best_idx)
    current_redisual = tf.gather(current_redisual, best_idx)
    current_assignment = tf.gather(current_assignment, best_idx)
    current_source_selection = tf.gather(current_source_selection, best_idx)
    current_residual_xy = tf.gather(current_residual_xy, best_idx)
    current_raw_results = [tf.gather(x, best_idx) for x in current_raw_results]
    #"""


    
    if try_drop and (tf.math.log(current_redisual)>-6.):#TEMP-7.):#drop up to 20%
        num_drop = tf.maximum(0, tf.shape(sources)[1]//5)#icp_fitting_func, icp_fitting_2d_batch_drop##, drop_residual_xy 
        num_drop = tf.minimum(2, num_drop)
        drop_sources, drop_matrix, drop_redisual, drop_assignment, drop_raw_results, drop_source_selection, drop_residual_xy = icp_fitting_2d_batch_confdrop_cost(targets, sources, all_invalid_mask, cost=st_cost_matrix, num_iter=num_fitting_iter,
                                                                          k = k, rz = rz, tx = tx, ty = ty, 
                                                                          l2_penalty=l2_penalty,
                                                                          num_harddrop=num_harddrop, num_softdrop=num_drop+num_softdrop)
    
        best_idx = tf.argmin(drop_redisual)
        drop_redisual = tf.gather(drop_redisual, best_idx)
        if (tf.math.log(current_redisual) - tf.math.log(drop_redisual))>1.0:
            #print("DROP!")
            current_redisual = drop_redisual
            current_sources = tf.gather(drop_sources, best_idx)
            current_matrix = tf.gather(drop_matrix, best_idx)
            current_assignment = tf.gather(drop_assignment, best_idx)
            current_source_selection = tf.gather(drop_source_selection, best_idx)
            current_residual_xy = tf.gather(drop_residual_xy, best_idx)

            current_raw_results = [tf.gather(x, best_idx) for x in drop_raw_results]
    
    if try_drop and (tf.math.log(current_redisual)>-3.):#TEMP>-5.):#still bad
        num_drop = tf.maximum(0, tf.shape(sources)[1]//3)#icp_fitting_func, icp_fitting_2d_batch_drop##, drop_residual_xy 
        num_drop = tf.minimum(4, num_drop)
        drop_sources, drop_matrix, drop_redisual, drop_assignment, drop_raw_results, drop_source_selection, drop_residual_xy = icp_fitting_2d_batch_confdrop_cost(targets, sources, all_invalid_mask, cost=st_cost_matrix, num_iter=num_fitting_iter,
                                                                          k = k, rz = rz, tx = tx, ty = ty, 
                                                                          l2_penalty=l2_penalty,
                                                                          num_harddrop=num_harddrop, num_softdrop=num_drop+num_softdrop)
    
        best_idx = tf.argmin(drop_redisual)
        drop_redisual = tf.gather(drop_redisual, best_idx)
        if (tf.math.log(current_redisual) - tf.math.log(drop_redisual))>1.0:
            #print("DROP!")
            current_redisual = drop_redisual
            current_sources = tf.gather(drop_sources, best_idx)
            current_matrix = tf.gather(drop_matrix, best_idx)
            current_assignment = tf.gather(drop_assignment, best_idx)
            current_source_selection = tf.gather(drop_source_selection, best_idx)
            current_residual_xy = tf.gather(drop_residual_xy, best_idx)

            current_raw_results = [tf.gather(x, best_idx) for x in drop_raw_results]
    
    
    
    current_source_selection = tf.cast(current_source_selection, tf.bool)

    #print(current_redisual)

    return current_redisual, current_matrix, current_sources, current_assignment, current_raw_results, current_source_selection, current_residual_xy




def tf_linear_sum_assignment(cost_matrix):
    def np_linear_sum_assignment(cost_matrix):
        return linear_sum_assignment(cost_matrix, maximize=False)[1].astype(np.int32)
    return tf.numpy_function(func=np_linear_sum_assignment,inp=[tf.cast(cost_matrix, tf.float32)],Tout=[tf.int32])

def tf_linear_sum_assignment_batch(cost_matrix):
    def np_linear_sum_assignment(cost_matrix):
        return np.array([linear_sum_assignment(cm, maximize=False)[1].astype(np.int32) for cm in cost_matrix])
    return tf.numpy_function(func=np_linear_sum_assignment,inp=[tf.cast(cost_matrix, tf.float32)],Tout=[tf.int32])





#"""
@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.bool),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, 4, 1], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.int32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              #tf.TensorSpec(shape=[], dtype=tf.float32),
                              #tf.TensorSpec(shape=[], dtype=tf.float32),
                              #tf.TensorSpec(shape=[], dtype=tf.float32),
                              #tf.TensorSpec(shape=[], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.int32),
                              tf.TensorSpec(shape=[], dtype=tf.int32),
                              )
             )#"""
def icp_fitting_2d_batch_confdrop_cost(targets, sources, conf_mask, cost, l2_penalty, num_iter=20,
                   k = 0., rz = 0., tx = 0., ty = 0., 
                   
                   #k_penalty=1.0, rz_penalty=1.0, tx_penalty=1.0, ty_penalty=1.0,
                   num_harddrop=0, num_softdrop=0):
    
    l2_penalty_matrix = tf.eye(4)[tf.newaxis,...] * l2_penalty
    
    def get_transmatrix(k, rz, tx, ty):
        """
        k : zoom ratio.
        rz : rotation.
        tx : x offset.
        ty : z offset
        shape [batch]
        
        returns:
            transmatrix with shape [batch, 3, 3]
        
        #TODO add penalty on k
        #add penalty on rotation angle
        l2 norm系パラメータ調整？初期値からの乖離の方がいいかも
        
        """
        exp_k = tf.math.exp(k)
        sin = tf.math.sin(rz)
        cos = tf.math.cos(rz)
        mat = tf.stack([[exp_k*cos, -exp_k*sin, exp_k*tx],
                        [exp_k*sin, exp_k*cos, exp_k*ty],
                        [tf.zeros_like(k), tf.zeros_like(k), tf.ones_like(k)]])
        mat = tf.transpose(mat, [2,0,1])
        return mat
        
    #def transform_points(transmatrix, points):
    #    x, y = tf.split(points, 2, axis=-1)
    #    xyones = tf.concat([x,y,tf.ones_like(x)], axis=-1)
    #    trans_points = tf.matmul(xyones, transmatrix, transpose_b=True)[...,:2]
    #    return trans_points
    
    #2->3 when use team
    def transform_points(transmatrix, points):
        x, y, team = tf.split(points, 3, axis=-1)
        xyones = tf.concat([x,y,tf.ones_like(x)], axis=-1)
        trans_points = tf.matmul(xyones, transmatrix, transpose_b=True)[...,:2]#do not warp for team axis
        trans_points = tf.concat([trans_points, team], axis=-1)
        return trans_points
    
    def get_derivative_at(k, rz, tx, ty, points):
        dev = 1e-5
        original = transform_points(get_transmatrix(k, rz, tx, ty), points)
        dxy_dk = (transform_points(get_transmatrix(k+dev, rz, tx, ty), points) - original) / dev
        dxy_drz = (transform_points(get_transmatrix(k, rz+dev, tx, ty), points) - original) / dev
        dxy_dtx = (transform_points(get_transmatrix(k, rz, tx+dev, ty), points) - original) / dev
        dxy_dty = (transform_points(get_transmatrix(k, rz, tx, ty+dev), points) - original) / dev
        return original, dxy_dk, dxy_drz, dxy_dtx, dxy_dty
    
    def normal_equation(X, Y, gn_rate=1.0, gd_rate=0.02):
        #l2_penalty = tf.stack([[[1.0 * k_penalty,
        #                         1.0 * rz_penalty,
        #                         1.0 * tx_penalty,
        #                         1.0 * ty_penalty]]])
        
        #not in use, 1.0/rz_penalty, 1.0/tx_penalty, 1.0/ty_penalty
        #Gaussian Neuton combind with Gradient descent
        XtX = tf.matmul(X, X, transpose_a=True) + l2_penalty_matrix
        
        #XtX_inv = np.linalg.inv(XtX)
        #XtX_inv = 1.0*XtX_inv + 0.5*np.eye(len(XtX))
        #results = np.dot(XtX_inv, np.dot(X.T, Y))
        
        results_neuton = tf.linalg.solve(XtX, tf.matmul(X, Y, transpose_a=True))
        #results_gaussian = tf.matmul(tf.tile(tf.eye(XtX.shape[1])[tf.newaxis,...], [XtX.shape[0],1,1]), tf.matmul(X, Y, transpose_a=True))
        #results_gaussian = tf.matmul(tf.eye(XtX.shape[0]), tf.matmul(X, Y, transpose_a=True))
        results = gn_rate*results_neuton# + gd_rate*results_gaussian
        
        return results
    
    def search_nearest_target(sources, targets, batch_size):
        #全てのsourceを使用するかどうか。linear sum assignmentがいいな。
        #行方向にsource,　列方向にtarget軸を持ってきて距離二乗値を入れる。
        dist_matrix = tf.reduce_sum((sources[:,:,tf.newaxis,:] - targets[:,tf.newaxis,:,:])**2, axis=-1)
        ##assignments = []
        ##assignment = tf.zeros((tf.shape(sources)[1]), tf.int32)
        ##for _ in tf.range(batch_size):
        ##    assignment = tf.reshape(tf_linear_sum_assignment(dist_matrix[i])[0], [tf.shape(sources)[1]])
        ##    assignments.append(assignment)
        #assignments = tf.tile(tf.range(tf.shape(sources)[1])[tf.newaxis,:], [tf.shape(sources)[0],1])
        
        #TODO  このアサインメントって距離二乗じゃなくて距離にすべきか？？？？
        #試してみる
        #dist_matrix = tf.math.sqrt(dist_matrix)
        assignments = tf_linear_sum_assignment_batch(dist_matrix)#[0]
        ##assignments = tf.stack(assignments)
        temp_targets = tf.gather(targets, assignments, batch_dims=1)
        #print(assignments)
        return temp_targets, assignments
    
    def search_nearest_target_drop(sources, targets, num_drop, additional_cost):
        #全てのsourceを使用するかどうか。linear sum assignment　全てアサインする必要が無いバージョン。
        #行方向にsource,　列方向にtarget軸を持ってきて距離二乗値を入れる。
        #target軸方向にゼロパッドすることで、アサインしにくいソースを捨てる。
        #TODO 最終で増やしても変わらないところを見定めるのにも使いたい。
        batch, num_targets, _ = tf.unstack(tf.shape(targets))
        batch, num_sources, _ = tf.unstack(tf.shape(sources))
        
        ##SORCES MUST BE SORTED BY CONFIDENCE
        ##num_pad = tf.minimum(num_lessconf, num_drop)
        num_can_drop = tf.minimum(num_sources, num_lessconf)
        no_cost_pad = tf.concat([tf.ones((batch,num_sources-num_can_drop,num_drop), tf.float32),
                                 tf.zeros((batch,num_can_drop,num_drop), tf.float32)],
                                 axis=1)
        
        dist_matrix = tf.reduce_sum((sources[:,:,tf.newaxis,:] - targets[:,tf.newaxis,:,:])**2, axis=-1)
        dist_matrix = additional_cost + dist_matrix
        #dist_matrix_pad = tf.pad(dist_matrix, [[0,0],[0,0],[0,num_pad]], "CONSTANT")
        dist_matrix_pad = tf.concat([dist_matrix, no_cost_pad], axis=-1)
        assignments = tf_linear_sum_assignment_batch(dist_matrix_pad)#[0]
        
        assignments = tf.reshape(assignments, [batch, num_sources])
        not_pad_mask = (assignments < num_targets)
        
        
        #assignments = tf.boolean_mask(assignments, not_pad_mask)
        assignments_valid = tf.reshape(tf.boolean_mask(assignments, not_pad_mask), [batch, num_sources-num_drop])
        temp_targets = tf.gather(targets, assignments_valid, batch_dims=1)
        #temp_targets = tf.reshape(temp_targets, [batch, num_targets, -1])
        #temp_targets_mask = tf.boolean_mask(temp_targets, not_pad_mask, axis=1)
        temp_sources = tf.boolean_mask(sources, not_pad_mask)
        temp_targets = tf.reshape(temp_targets, [batch, num_sources-num_drop, -1])
        temp_sources = tf.reshape(temp_sources, [batch, num_sources-num_drop, -1])
        #print(assignments_valid.shape)
        #print(assignments.shape)
        """
        source_idx = tf.tile(tf.range(num_sources)[tf.newaxis,:], [batch_size, 1])
        target_idx = tf.tile(tf.range(num_targets)[tf.newaxis,:], [batch_size, 1])
        pad_mask = assignments >= num_targets
        assigned_idx_source
        not_assigned_idx_source
        assigned_idx_target
        not_assigned_idx_target
        
        target_idx = tf.cast(tf.range(t_num_points)[tf.newaxis,:,tf.newaxis], tf.float32)
        assigned_index = tf.cast(assignments, tf.float32)[:,tf.newaxis,:]
        #dev_index = tf.boolean_mask(tf.tile(base_index[:,:,0],[batch,1]), tf.reshape(tf.reduce_min((base_index - assigned_index)**2, axis=-1)>0.1, [batch, t_num_points]))
        dev_index = tf.boolean_mask(tf.tile(base_index[:,:,0],[batch,1]), tf.reshape(tf.reduce_all(base_index != assigned_index, axis=-1), [batch, t_num_points]))
        not_assigned_idx_target = tf.reshape(tf.cast(dev_index, tf.int32), [batch, -1])
        targets_remain = tf.gather(targets, not_assigned_idx_target, batch_dims=1)
        """
        
        return temp_targets, temp_sources, assignments, assignments_valid, not_pad_mask
    
    def final_search_nearest_target_drop(sources, targets, num_drop, additional_cost):
        #全てのsourceを使用するかどうか。linear sum assignment　全てアサインする必要が無いバージョン。
        #行方向にsource,　列方向にtarget軸を持ってきて距離二乗値を入れる。
        #target軸方向にゼロパッドすることで、アサインしにくいソースを捨てる。
        #TODO 最終で増やしても変わらないところを見定めるのにも使いたい。
        """
        dist_matrix = tf.reduce_sum((sources[:,:,tf.newaxis,:] - targets[:,tf.newaxis,:,:])**2, axis=-1)
        dist_matrix_pad = tf.pad(dist_matrix, [[0,0],[0,0],[0,num_drop]], "CONSTANT")
        assignments = tf_linear_sum_assignment_batch(dist_matrix_pad)[0]
        temp_targets = tf.gather(targets, assignments, batch_dims=1)
        not_pad_mask = (assignments < num_targets)
        temp_targets = tf.boolean_mask(temp_targets, not_pad_mask, axis=1)
        temp_sources = tf.boolean_mask(sources, not_pad_mask, axis=1)
        """
        batch, num_targets, _ = tf.unstack(tf.shape(targets))
        batch, num_sources, _ = tf.unstack(tf.shape(sources))

        targets_assigned, sources_assigned, assigned_idx_target, assigned_idx_target_valid, not_pad_mask = search_nearest_target_drop(sources, targets, num_drop, additional_cost)

        not_pad_mask = (assigned_idx_target < num_targets)
        pad_mask = (assigned_idx_target >= num_targets)
        source_idx = tf.tile(tf.range(num_sources)[tf.newaxis,:], [batch, 1])
        #target_idx = tf.tile(tf.range(num_targets)[tf.newaxis,:], [batch, 1])
        #print(source_idx.shape, not_pad_mask.shape)
        assigned_idx_source = tf.boolean_mask(source_idx, not_pad_mask)#, axis=1)
        not_assigned_idx_source = tf.boolean_mask(source_idx, pad_mask)#, axis=1)
        assigned_idx_source = tf.reshape(assigned_idx_source, [batch, num_sources-num_drop])
        not_assigned_idx_source = tf.reshape(not_assigned_idx_source, [batch, num_drop])        
        #assigned_idx_target = 
        #not_assigned_idx_target
        
        target_idx = tf.cast(tf.range(num_targets)[tf.newaxis,:,tf.newaxis], tf.float32)
        assigned_index = tf.cast(assigned_idx_target, tf.float32)[:,tf.newaxis,:]
        dev_index = tf.boolean_mask(tf.tile(target_idx[:,:,0],[batch,1]), tf.reshape(tf.reduce_all(target_idx != assigned_index, axis=-1), [batch, num_targets]))
        not_assigned_idx_target = tf.reshape(tf.cast(dev_index, tf.int32), [batch, -1])
        
        sources_remain = tf.gather(sources, not_assigned_idx_source, batch_dims=1)
        targets_remain = tf.gather(targets, not_assigned_idx_target, batch_dims=1)
        #print(sources_remain)
        #print(targets_remain)
        return targets_assigned, sources_assigned, targets_remain, sources_remain, assigned_idx_target_valid, assigned_idx_source, not_assigned_idx_target, not_assigned_idx_source
    
    def get_current_location(k, rz, tx, ty, sources):
        trans_matrix = get_transmatrix(k, rz, tx, ty)
        current = transform_points(trans_matrix, sources)
        return current
    
    
    
    
    batch, num_targets, _ = tf.unstack(tf.shape(targets))
    batch, num_sources, _ = tf.unstack(tf.shape(sources))
    
    cost = tf.tile(cost[tf.newaxis,:,:],[batch,1,1])
    num_lessconf = tf.reduce_sum(1 - tf.cast(conf_mask,tf.int32))
    

    sources_original = sources
    batch, num_points, _ = tf.unstack(tf.shape(sources))
    
    #k = 0.#0 * tf.ones((batch), tf.float32)#expで取る。ネガティブでんし。
    #rz = 0.#0 * tf.ones((batch), tf.float32)
    #tx = 0.#0 * tf.ones((batch), tf.float32)
    #ty = 0.#0 * tf.ones((batch), tf.float32)
    #limit_updates = 1.0 * tf.ones((1,1),tf.float32)
    ##not using
    ##limit_updates = tf.stack([[1.0/k_penalty, 1.0/rz_penalty, 1.0/tx_penalty, 1.0/ty_penalty]])
    #limit_updates = tf.stack([[1.0, 1.0, 1.0, 1.0]])/rz_penalty
    ##limit_updates = tf.tile(limit_updates, [batch, 1])
    ##num_top = num_points-(num_points//10)

    drop_during_iter = num_harddrop + num_softdrop
    drop_during_iter = tf.minimum(num_lessconf, drop_during_iter)
    drop_final = tf.minimum(num_lessconf, num_harddrop)
    # CONFIDENCE THresh MUST BE ENOUGH LARGE. if too small, even hard drop will not work 
    for i in range(num_iter):
        
        current = get_current_location(k, rz, tx, ty, sources)
        temp_targets, _, _, _, not_pad_mask = search_nearest_target_drop(current, targets, drop_during_iter, cost)
        
        #temp_targets, _ = search_nearest_target(current, targets, batch)
        #print(_)
        temp_sources = tf.boolean_mask(sources, not_pad_mask)
        temp_sources = tf.reshape(temp_sources, [batch, num_sources-drop_during_iter, -1])
        ##temp_targets, _ = search_nearest_target(sources, targets, batch_size)
        currents, dxy_dk, dxy_rz, dxy_dtx, dxy_dty = get_derivative_at(k, rz, tx, ty, temp_sources)
        delta = temp_targets - currents
        """
        
        current = get_current_location(k, rz, tx, ty, sources)
        temp_targets, _ = search_nearest_target(current, targets, batch_size)
        ##temp_targets, _ = search_nearest_target(sources, targets, batch_size)
        currents, dxy_dk, dxy_rz, dxy_dtx, dxy_dty = get_derivative_at(k, rz, tx, ty, sources)
        delta = temp_targets-currents        
        #"""
        
        #nonlinear weight by distance
        """
        distance = tf.math.sqrt(tf.reduce_sum(delta[...,:2]**2, axis=-1, keepdims=True))
        weight = distance/tf.reduce_mean(distance, axis=1, keepdims=True)
        weight = 1.0 / tf.maximum(weight, 1.0)#平均より遠い場合はウェイト下げる
        
        #and条件ある方がいいかな。0.1distance以上とか。 あと、今フラットさちりなので、ノンリニア化する。
        b = tf.reshape(delta * weight, [-1, num_points*3, 1])#x-y-team flatten
        a = tf.stack([dxy_dk, dxy_rz, dxy_dtx, dxy_dty], axis=-1)
        a = tf.reshape(a * weight[...,tf.newaxis], [-1, num_points*3, 4])
        """
        #a = tf.reshape(a, [-1, num_points*3, 4])
        b = tf.reshape(delta, [-1, (num_points-drop_during_iter)*3, 1])#x-y-team flatten
        a = tf.stack([dxy_dk, dxy_rz, dxy_dtx, dxy_dty], axis=-1)
        a = tf.reshape(a, [-1, (num_points-drop_during_iter)*3, 4])
        
        
        
        """
        # use top k
        #　この時点でtopkとるよりも、linear assignmentのタイミングで、これ除けば楽なのに…。というやつを見つけるほうがいい気がする。
        # やっぱりランダムドロップ？？？
        distance = tf.reduce_sum(delta[...,:2]**2, axis=-1)
        values, indices = tf.math.top_k(-distance, k=num_top)
        delta = tf.gather(delta, indices, batch_dims=1)
        b = tf.reshape(delta, [-1, num_top*3, 1])#x-y-team flatten
        a = tf.stack([dxy_dk, dxy_rz, dxy_dtx, dxy_dty], axis=-1)
        a = tf.gather(a, indices, batch_dims=1)
        a = tf.reshape(a, [-1, num_top*3, 4])
        """
        
        #delta_dist = tf.reduce_sum(delta**2, axis=-1, keepdims=True)
        #delta_base_weight = tf.reshape(tf.tile(tf.reduce_mean(delta_dist)/(delta_dist)+1e-20, [1,2]), [num_points*2, 1])
        #a *= delta_base_weight
        #b *= delta_base_weight
        
        #updates = tf.linalg.lstsq(a, b, l2_regularizer=0.01, fast=True)#batch, 4, 1
        updates = normal_equation(a,b)#[:,:,0]
        updates = tf.reshape(updates, [batch, 4])
        #print(updates)
        #clip update
        #max_update = tf.math.abs(tf.reduce_max(updates, axis=2, keepdims=True))
        
        ##updates_ratio = tf.reduce_min(limit_updates / (tf.math.abs(updates)+1e-12), axis=1, keepdims=True)
        ##updates = updates * tf.minimum(updates_ratio, tf.ones_like(updates_ratio))
        
        #print(updates)
        
        k = k + updates[:,0]
        rz = rz + updates[:,1]
        tx = tx + updates[:,2]
        ty = ty + updates[:,3]
        
    trans_matrix = get_transmatrix(k, rz, tx, ty)
    trans_sources = transform_points(trans_matrix, sources)
    targets_assigned, sources_assigned, targets_remain, sources_remain, assigned_idx_target, assigned_idx_source, not_assigned_idx_target, not_assigned_idx_source = final_search_nearest_target_drop(trans_sources, targets, drop_during_iter, cost)
    
    #B, N, M and B, n -> B, n, M
    #B, n, M and B, m -> B, n, m
    cost_remain = tf.gather(cost, not_assigned_idx_source, batch_dims=1)
    cost_remain = tf.transpose(cost_remain, [0,2,1])
    cost_remain = tf.gather(cost_remain, not_assigned_idx_target, batch_dims=1)
    cost_remain = tf.transpose(cost_remain, [0,2,1])
    
    targets_reassigned, sources_reassigned, _as, assignments_valid, not_pad_mask = search_nearest_target_drop(sources_remain, targets_remain, drop_final, cost_remain)
    ###_, remain_assignments = search_nearest_target(sources_remain, targets_remain, batch)
    ###remain_assignments = tf.gather(not_assigned_idx_target, remain_assignments, batch_dims=1)
    remain_assignments = tf.gather(not_assigned_idx_target, assignments_valid, batch_dims=1)
    
    
    #mean 間違ってる…。
    #residual_assigned = tf.reduce_mean((targets_assigned - sources_assigned)**2, axis=[1,2])
    residual_assigned = tf.reduce_mean(tf.reduce_sum((targets_assigned - sources_assigned)**2, axis=2), axis=1)
    residual_assigned_xy = tf.reduce_sum((targets_assigned[...,:2] - sources_assigned[...,:2])**2, axis=2)
    residual_reassigned_xy = tf.reduce_sum((targets_reassigned[...,:2] - tf.reshape(sources_reassigned[...,:2], [batch,-1,2]))**2, axis=2)
    
    final_assignments_unsort = tf.concat([assigned_idx_target, remain_assignments], axis=1)
    argsort_source_all = tf.argsort(tf.concat([assigned_idx_source, not_assigned_idx_source], axis=1), axis=1)
    not_assigned_idx_source = tf.reshape(tf.boolean_mask(not_assigned_idx_source, not_pad_mask),[batch,-1])
    argsort_source = tf.argsort(tf.concat([assigned_idx_source, not_assigned_idx_source], axis=1), axis=1)
    source_selection = tf.concat([tf.ones_like(assigned_idx_source), tf.cast(not_pad_mask,tf.int32)], axis=1)
    residual_xy = tf.concat([residual_assigned_xy, residual_reassigned_xy], axis=1)
    final_assignments = tf.gather(final_assignments_unsort, argsort_source, batch_dims=1)
    source_selection = tf.gather(source_selection, argsort_source_all, batch_dims=1)
    residual_xy = tf.gather(residual_xy, argsort_source, batch_dims=1)
    
    residual = residual_assigned
    raw_results = [k, rz, tx, ty]
    #print(trans_matrix)
    return trans_sources, trans_matrix, residual, final_assignments, raw_results, source_selection, residual_xy






if __name__ == "__main__":
    
    targets = tf.constant([[0,0],
                           [1,0.1],
                           [2,0],
                           [3,0.25],
                           [4,0],
                           [2,2],
                           [1.3,1.6],
                           [1.3,-1.3]], tf.float32)
    sources = targets[:6]*0.5 + 0.00001-0.5
    
    targets_team = tf.constant([[0.05],
                           [1.0],
                           [2.0],
                           [3.25],
                           [4.0],
                           [1.2],
                           [3.3],
                           [1.2],
                           ], tf.float32)
    sources_team = targets_team[:6]
    # must be sorted
    confidence = tf.constant([0.3,0.2,0.15,0.15,0.05,0.04], tf.float32)
        
        #current_redisual, current_matrix, current_sources, current_assignment, current_raw_results
    #results = random_icp_fitting_batch(targets, 
    #                   sources,
    #                   confidence,
    #                   targets_team,
    #                   sources_team,   
    #                   num_trial=2,
    #                   )
    confidence_threshold = 0.1
    num_harddrop = 1
                       
    results = random_icp_fitting_batch_drop_cost(targets[:6], 
                       sources,
                       confidence,
                       targets_team[:6],
                       sources_team,   
                       st_cost_matrix=tf.zeros((len(sources_team), 6), tf.float32),
                       num_trial=2,
                       #use_provided_params=True,
                       #txy_params = tf.constant([[0,0,100.],[0,0,100.]], tf.float32),
                       confidence_threshold=confidence_threshold,
                       num_harddrop=num_harddrop)
    print("OK", results[0])
    print(results[2])
    print(results[3])
    print(results[-1])
    plt.scatter(targets[:,0],targets[:,1])
    plt.scatter(results[2][:,0],results[2][:,1])
    plt.show()
    raise Exception("e")
    
    """
    matrix = tf.constant([[0.5,0.2,0.6,0.8],
                          [0.1,0.3,0.4,0.2],
                          [0.6,0.6,0.2,0.3]])
    assignment = tf_linear_sum_assignment(matrix)
    print(assignment)
    """
    
    """
    targets = tf.constant([[0,0],
                           [1,0.1],
                           [2,0],
                           [3,0.25],
                           [4,0],
                           [2,2],
                           [1.3,1.6],
                           [1.3,-1.3]], tf.float32)
    sources = targets[:5]*0.8 + 0.6
    
    #trans_sources, trans_matrix, residual, assigned_targets = icp_fitting_2d(targets, sources, num_iter=20, rz=0.7)
    #print(trans_sources, residual, assigned_targets)
    
    print(icp_fitting_2d(targets, sources))

    random_icp_fitting(targets, sources, num_trial=30)
    """
    
    
    print("-"*10)
    targets = tf.constant([[0,0],
                           [1,0],
                           [2,0],
                           [3,0],
                           [4,0],
                           [1,1],
                           [2,1],
                           [3,1],
                           ], tf.float32)
    sources = targets[:5]*0.2 + 0.1
    
    trans_sources, trans_matrix, residual, final_assignment, raw_results = icp_fitting_2d(targets, sources, num_iter=1)
    plt.scatter(trans_sources[:,0],trans_sources[:,1])
    plt.scatter(targets[:,0],targets[:,1])
    plt.show()
    trans_sources, trans_matrix, residual, final_assignment, raw_results = icp_fitting_2d(targets, sources, num_iter=2)
    plt.scatter(trans_sources[:,0],trans_sources[:,1])
    plt.scatter(targets[:,0],targets[:,1])
    plt.show()
    trans_sources, trans_matrix, residual, final_assignment, raw_results = icp_fitting_2d(targets, sources, num_iter=3)
    plt.scatter(trans_sources[:,0],trans_sources[:,1])
    plt.scatter(targets[:,0],targets[:,1])
    plt.show()

    raise Exception("e")
    import time
    s = time.time()
    for i in range(10):
        results = random_icp_fitting_batch(targets, sources, num_trial=30)
        print("TIME", time.time()-s)
        
    print(results["residual"])
    print(results["final_assignment"])
    s = results["trans_sources"].numpy()
    plt.scatter(s[:,0],s[:,1])
    plt.scatter(targets[:,0],targets[:,1])
    plt.show()
    
    #"""
    for _ in range(10):
        print("-"*10)
        targets = tf.constant([[0,0],
                               [1,0.1],
                               [2,0],
                               [3,0.25],
                               [4,0],
                               [2,2],
                               [1.3,1.6],
                               [1.3,-1.3]], tf.float32)
        sources = targets[:5]*0.8 + 0.6
    
        random_icp_fitting(targets, sources, num_trial=30)
        
    print(time.time()-s)
    #"""
