# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 22:06:26 2021

@author: kmat

"""
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

class Tracker(object):
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.info = {}#{"TrackingID1":{"NUM_FRAME_1":"PLAYER_ID", "NUM_FRAME_2": "PLAYER_ID"},
                     #"TrackingID2":{}}
        self.tracking_speed = {}#"TrackingID1":[vx,vy]
        self.weights = {}
        self.iou_weights = {}
        self.ious = {}
        self.box_to_id ={}# {"PLAYID_VIEW_FRAME":[3,1,5,1,7,9],
                          #  "PLAYID_VIEW_FRAME":[]}
        self.box_history = {}                  
        self.box_history_conf = {}                  
        self.box_history_loc = {}                  
        self.box_history_mask = {}
        self.current_id = 0
        
    def set_tracking_data(self, df_tracking, frames):
        df_tracking = df_tracking.copy()
        min_frame = np.array(frames).min()
        max_frame = np.array(frames).max()
        df_tracking = df_tracking[df_tracking["frame"]>=min_frame]
        df_tracking = df_tracking[df_tracking["frame"]<= max_frame]
        players = df_tracking["player"].unique()
        self.players = list(players)
        self.tracking_matrix_gt = np.zeros((len(players), len(frames), 4), np.float32)
        for i, p in enumerate(players):
            df = df_tracking[df_tracking["player"]==p]
            fs = df["frame"].values.reshape(-1).astype(int) - 1# frames starts from zero
            xs = df["x"].values.reshape(-1).astype(float)/20.0
            ys = df["y"].values.reshape(-1).astype(float)/20.0
            ts = np.ones_like(ys) * float("H" in p)
            self.tracking_matrix_gt[i, fs, 0] = xs
            self.tracking_matrix_gt[i, fs, 1] = ys
            self.tracking_matrix_gt[i, fs, 2] = ts
            self.tracking_matrix_gt[i, fs, 3] = 1.#mask
        
    def ious_to_weight(self, ious):
        return np.maximum(0.8-ious, 0.01/2)    

    def get_tracking_box_id(self, game_play, view, frame, box_idx):
        return self.box_to_id["{}_{}_{}".format(game_play, view, frame)][box_idx]

    def add_tracking_box_id(self, game_play, view, frame, new_track_id, players, weight, ious):
        ious_weight = self.ious_to_weight(ious)
        self.box_to_id["{}_{}_{}".format(game_play, view, frame)] = new_track_id
        for t_id, pl, w, iou, iou_w in zip(new_track_id, players, weight, ious, ious_weight):
            if t_id in self.info.keys():                
                self.info[t_id].update({frame: pl})  
                self.weights[t_id].update({frame: w})
                self.iou_weights[t_id].update({frame: iou_w})
                #self.ious[t_id].update({frame: iou})
            else:
                self.info[t_id] = {frame: pl}
                self.weights[t_id] = {frame: w}
                self.iou_weights[t_id] = {frame: iou_w}
                #self.ious[t_id] = {frame: iou}
    
    def set_box_history(self, game_play, view, frame, boxes, loc, conf=None, icp_errors=None):
        if icp_errors is None:
            icp_error = 1.0
        else:
            icp_error = np.exp(icp_errors).mean()
        self.box_history["{}_{}_{}".format(game_play, view, frame)] = boxes
        self.box_history_loc["{}_{}_{}".format(game_play, view, frame)] = loc
        self.box_history_mask["{}_{}_{}".format(game_play, view, frame)] = np.ones((len(boxes),)) / icp_error
        if conf is None:
            conf = np.ones((len(boxes),), np.float32)
        self.box_history_conf["{}_{}_{}".format(game_play, view, frame)] = conf        
        
    def add_box_history_and_id(self, game_play, view, frame, boxes, assigned_id, ious, conf=0.25):
        """
        """
        iou_mat = get_iou_matrix(self.box_history["{}_{}_{}".format(game_play, view, frame)], boxes)
        ious_mask = np.max(iou_mat, axis=0) < 0.6
        boxes = boxes[ious_mask]
        assigned_id = assigned_id[ious_mask]
        ious = ious[ious_mask]
        
        ious_weight = self.ious_to_weight(ious)
        self.box_history["{}_{}_{}".format(game_play, view, frame)] = np.concatenate([self.box_history["{}_{}_{}".format(game_play, view, frame)], boxes], axis=0)
        self.box_history_loc["{}_{}_{}".format(game_play, view, frame)] = np.concatenate([self.box_history_loc["{}_{}_{}".format(game_play, view, frame)], np.zeros((len(boxes),2))], axis=0)
        self.box_history_mask["{}_{}_{}".format(game_play, view, frame)] = np.concatenate([self.box_history_mask["{}_{}_{}".format(game_play, view, frame)], np.zeros((len(boxes),))], axis=0)
        #if conf is None:
        conf = conf * np.ones((len(boxes),), np.float32)
        self.box_history_conf["{}_{}_{}".format(game_play, view, frame)] = np.concatenate([self.box_history_conf["{}_{}_{}".format(game_play, view, frame)], conf], axis=0)
        self.box_to_id["{}_{}_{}".format(game_play, view, frame)] = np.concatenate([self.box_to_id["{}_{}_{}".format(game_play, view, frame)], assigned_id], axis=0)
        for t_id, iou, iou_w in zip(assigned_id, ious, ious_weight):
            self.info[t_id].update({frame: "invalid_player"})
            self.weights[t_id].update({frame: 0.0})
            self.iou_weights[t_id].update({frame: iou_w})
            #self.ious[t_id].update({frame: iou})
        
    def remove_single_frame_box(self, game_play, view):
        for t_id, assignment in self.info.items():
            if len(assignment)==1:
                frame = list(assignment.keys())[0]
                b2id = self.box_to_id["{}_{}_{}".format(game_play, view, frame)]
                survive_mask = (np.array(b2id) != t_id)
                self.box_to_id["{}_{}_{}".format(game_play, view, frame)] = self.box_to_id["{}_{}_{}".format(game_play, view, frame)][survive_mask]
                self.box_history["{}_{}_{}".format(game_play, view, frame)] = self.box_history["{}_{}_{}".format(game_play, view, frame)][survive_mask]
                self.box_history_conf["{}_{}_{}".format(game_play, view, frame)] = self.box_history_conf["{}_{}_{}".format(game_play, view, frame)][survive_mask]
    
    def remake_box_prediction(self, game_play, view):
        self.remove_single_frame_box(game_play, view)
        df_all = []
        for key in self.box_history.keys():
            game_play, view, frame = key.rsplit("_",2)
            boxes_tlbr = self.box_history[key]
            boxes_hw = boxes_tlbr[:,2:4] - boxes_tlbr[:,:2]
            boxes_tlhw = np.concatenate([boxes_tlbr[:,:2], boxes_hw], axis=-1)
            boxes_conf = self.box_history_conf[key]
            
            df = pd.DataFrame(boxes_tlhw, columns=["top", "left", "height", "width"])
            df['conf'] = boxes_conf
            df['view'] = view
            df['game_play'] = game_play
            df['frame'] = frame
            df_all.append(df)
        df_all = pd.concat(df_all, axis=0)
        return df_all     
    
    def add_tracking_speed(self, track_boxes_0, track_boxes_1, track_id, new_ids, update_ratio=0.5):
        for t_id in new_ids:
            self.tracking_speed[t_id] = np.array([0.0, 0.0])
        if track_boxes_0 is not None:
            cx_0 = (track_boxes_0[:,1] + track_boxes_0[:,3])/2.0
            cy_0 = (track_boxes_0[:,0] + track_boxes_0[:,2])/2.0
            cx_1 = (track_boxes_1[:,1] + track_boxes_1[:,3])/2.0
            cy_1 = (track_boxes_1[:,0] + track_boxes_1[:,2])/2.0
            vx = cx_1 - cx_0
            vy = cy_1 - cy_0
            for i,t_id in enumerate(track_id):
                self.tracking_speed[t_id] = self.tracking_speed[t_id] * (1.0-update_ratio) + np.array([vx[i],vy[i]]) * update_ratio
        
    def estimate_current_frame_box(self, previous_boxes, previous_tracked_id):
        speeds = np.array([self.tracking_speed[t_id] for t_id in previous_tracked_id])
        vyvxvyvx = np.concatenate([speeds, speeds], axis=-1)[:,::-1]
        return previous_boxes + vyvxvyvx#should use clip?
    
    #def predict_and_add(self, game_play, view, frame, assigned_player, current_boxes, weight=1.0, conf=None):
    #    self.set_box_history(game_play, view, frame, current_boxes, conf=conf)
    def predict_and_add(self, game_play, view, frame, assigned_player, current_boxes, locations, weight=1.0, conf=None):
        self.set_box_history(game_play, view, frame, current_boxes, locations, conf=conf)
        if len(current_boxes)>0:
            new_track_id = self.current_id + np.arange(len(current_boxes))#dafault
            if self.previous_boxes is not None:
                prev_boxes = self.estimate_current_frame_box(self.previous_boxes, self.previous_tracked_id)
                p_idx, c_idx = self._predict_by_iou(prev_boxes, current_boxes)
                tracked_id = self.get_tracking_box_id(game_play, view, self.previous_frame, p_idx)
                new_track_id[c_idx] = tracked_id
                not_assigned = [i for i in range(len(current_boxes)) if i not in c_idx]
                num_new_track = len(current_boxes) - len(c_idx)
                new_ids = self.current_id + np.arange(num_new_track)
                new_track_id[not_assigned] = new_ids
                
                p_track = self.previous_boxes[p_idx]
                c_track = current_boxes[c_idx]
                
                
                #end_id = np.aray(list(set(self.pprevious_tracked_id) - set(self.previous_tracked_id)))
                    
                
                #new_track_id[not_assigned] = np.arange(num_new_track)
            else:
                p_track = None
                c_track = None
                tracked_id = []
                num_new_track = len(current_boxes)
                new_ids = self.current_id + np.arange(num_new_track)
            #"""    
            #TODO
            if num_new_track>0 and len(self.pprevious_tracked_id)>0:
                #end_id = list(set(self.pprevious_tracked_id) - set(self.previous_tracked_id))
                #if len(end_id)>0:
                not_assigned_boxes = current_boxes[not_assigned]
                candidate_pprevious_boxes = []
                candidate_track_ids = []
                
                for i, tr_id in enumerate(self.pprevious_tracked_id):
                    if tr_id in self.previous_tracked_id:
                        continue
                    candidate_track_ids.append(tr_id)
                    candidate_pprevious_boxes.append(self.pprevious_boxes[i])
                candidate_pprevious_boxes = np.array(candidate_pprevious_boxes)
                if len(candidate_track_ids)>0:
                    p_idx, c_idx = self._predict_by_iou(candidate_pprevious_boxes, not_assigned_boxes)
                    lost_boxes = candidate_pprevious_boxes[p_idx]
                    emerge_boxes = not_assigned_boxes[c_idx]
                    average_boxes = (lost_boxes + emerge_boxes)/2
                    re_assigned_id = np.array(candidate_track_ids)[p_idx]
                    ###self.add_box_history_and_id(game_play, view, frame-1, average_boxes, re_assigned_id)
                    
                    #re_assign = np.array(not_assigned)[c_idx]
                    #new_track_id[re_assign] = re_assigned_id                
               
            self.add_tracking_speed(p_track, c_track, tracked_id, new_ids)
            
            self.add_tracking_box_id(game_play, view, frame, new_track_id, assigned_player, weight)
            self.pprevious_boxes = self.previous_boxes
            self.pprevious_frame = self.previous_frame
            self.pprevious_tracked_id = self.previous_tracked_id
            self.previous_boxes = current_boxes
            self.previous_frame = frame
            self.previous_tracked_id = new_track_id
            self.current_id += num_new_track
            
        else:
            self.previous_boxes = None
            self.previous_frame = None
            self.previous_tracked_id = []
            
    def reassign_player_label(self, game_play, view, frame, frame_sigma=5.0):
        track_ids = self.box_to_id["{}_{}_{}".format(game_play, view, frame)]
        num_box = len(track_ids)
        #new_assignments = []
        all_players_score = {}
        for box_idx, track_id in enumerate(track_ids):
            track_info = self.info[track_id]
            track_weights = self.weights[track_id]
            players_score = {}    
            frame_list = list(track_info.keys())
            weights = np.array(list(track_weights.values()))
            track_iou_weights = self.iou_weights[track_id]
            frame_idx = frame_list.index(frame)
            
            # iou-based decay
            iou_weights = np.array(list(track_iou_weights.values()))
            iou_weights = np.cumsum(iou_weights)
            iou_weights = np.abs(iou_weights - iou_weights[frame_idx])
            decay = 2.5**(-iou_weights)*2
            #dev_frames = np.abs(np.array(frame_list)-frame)
            #simple_decay = np.exp(-dev_frames/frame_sigma)
            #decay=simple_decay
            scores = weights * decay
 
            for score, player in zip(scores, track_info.values()):
                if player == "invalid_player":
                    continue                
                if player in players_score.keys():
                    players_score[player] += score
                else:
                    players_score[player] = score

            
            for p, s in players_score.items():
                if p in all_players_score.keys():
                    all_players_score[p][box_idx] = s
                else:
                    scores_list = np.zeros((num_box))
                    scores_list[box_idx] = s
                    all_players_score[p] = scores_list
        cost_matrix = np.array(list(all_players_score.values())).T
        player_labels = list(all_players_score.keys())
        boxes = self.box_history["{}_{}_{}".format(game_play, view, frame)]
        confs = self.box_history_conf["{}_{}_{}".format(game_play, view, frame)]
        num_pred = len(confs)
        num_player = len(player_labels)
        if num_pred > num_player:
            argsort = np.argsort(confs)[::-1]#sort from high score
            argsort = argsort[:num_player]
            boxes = boxes[argsort]
            confs = confs[argsort]
            cost_matrix = cost_matrix[argsort]            
        
        if cost_matrix.shape[0]>cost_matrix.shape[1]:
            raise Exception("tracked player is fewer than boxes. why?")
            
        
        _, players_idx = linear_sum_assignment(cost_matrix, maximize=True)
        new_assignments = [player_labels[idx] for idx in players_idx] 
        return boxes, confs, new_assignments

        
    def _get_iou_matrix(self, boxes_0, boxes_1):
        """
        boxes [N0, 4(top left bottom right)], [N1, 4(top left bottom right)]
        #left", "width", "top", "height"
        return:
            iou_matrix [N0, N1]
        """
        boxes_0 = boxes_0[:,np.newaxis,:]
        boxes_1 = boxes_1[np.newaxis,:,:]
        tops = np.maximum(boxes_0[:,:,0], boxes_1[:,:,0])
        lefts = np.maximum(boxes_0[:,:,1], boxes_1[:,:,1])
        bottoms = np.minimum(boxes_0[:,:,2], boxes_1[:,:,2])
        rights = np.minimum(boxes_0[:,:,3], boxes_1[:,:,3])
        intersection = np.maximum(bottoms-tops,0) * np.maximum(rights-lefts,0)
        area_0 = (boxes_0[:,:,2] - boxes_0[:,:,0]) * (boxes_0[:,:,3] - boxes_0[:,:,1])
        area_1 = (boxes_1[:,:,2] - boxes_1[:,:,0]) * (boxes_1[:,:,3] - boxes_1[:,:,1])
        union = area_0 + area_1 - intersection
        iou_matrix = intersection/(union+1e-7)
        return iou_matrix
    
    def _get_ciou_matrix(self, boxes_0, boxes_1):
        boxes_0 = boxes_0[:,np.newaxis,:]
        boxes_1 = boxes_1[np.newaxis,:,:]
        
        enclosing_top = np.minimum(boxes_0[:,:,0], boxes_1[:,:,0])
        enclosing_left = np.minimum(boxes_0[:,:,1], boxes_1[:,:,1])
        enclosing_bottom = np.maximum(boxes_0[:,:,2], boxes_1[:,:,2])
        enclosing_right = np.maximum(boxes_0[:,:,3], boxes_1[:,:,3])
        intersection_top = np.maximum(boxes_0[:,:,0], boxes_1[:,:,0])
        intersection_left =  np.maximum(boxes_0[:,:,1], boxes_1[:,:,1])
        intersection_bottom = np.minimum(boxes_0[:,:,2], boxes_1[:,:,2])
        intersection_right = np.minimum(boxes_0[:,:,3], boxes_1[:,:,3])
        
        b0_width = boxes_0[:,:,3] - boxes_0[:,:,1]
        b0_height = boxes_0[:,:,2] - boxes_0[:,:,0]
        b1_width = boxes_1[:,:,3] - boxes_1[:,:,1]
        b1_height = boxes_1[:,:,2] - boxes_1[:,:,0]
    
        # distance to calculate DIoU.   
        box_dist = (((boxes_0[:,:,0]+boxes_0[:,:,2])-(boxes_1[:,:,0]+boxes_1[:,:,2]))/2)**2 + (((boxes_0[:,:,1]+boxes_0[:,:,3])-(boxes_1[:,:,1]+boxes_1[:,:,3]))/2)**2
        diagonal_dist = (enclosing_top - enclosing_bottom)**2 + (enclosing_left - enclosing_right)**2
        
        b0_area = b0_width * b0_height
        b1_area = b1_width * b1_height
        intersection_area = np.maximum(intersection_bottom-intersection_top,0) * np.maximum(intersection_right-intersection_left,0)
        union_area = b0_area + b1_area - intersection_area
        iou = intersection_area / (union_area + 1e-7)#tf.math.divide_no_nan(intersection_area, union_area)
        
        #enclosing_area = (enclosing_bottom + enclosing_top) * (enclosing_right + enclosing_left)
        #giou = iou - (enclosing_area - union_area)/enclosing_area
        #giou_loss = 1.0 - giou
        
        diou = iou - box_dist / (diagonal_dist + 1e-7)#tf.math.divide_no_nan)
        # completed iou considers aspect ratio
        b0_aspect_ratio = b0_width / (b0_height + 1e-7)
        b1_aspect_ratio = b1_width / (b1_height + 1e-7)
        v = ((np.arctan(b0_aspect_ratio)
              - np.arctan(b1_aspect_ratio)) * 2 / np.pi) ** 2
        alpha = v/ (1 - iou + v)
    
        ciou = diou - alpha * v
        return ciou
    
    def _predict_by_iou(self, boxes_0, boxes_1, return_iou=False, iou_threshold=None):
        iou_threshold = iou_threshold or self.iou_threshold
        iou_matrix = self._get_iou_matrix(boxes_0, boxes_1)
        b0_idx, b1_idx = linear_sum_assignment(iou_matrix, maximize=True)
        iou_scores = iou_matrix[b0_idx, b1_idx]
        mask = iou_scores > iou_threshold
        b0_idx = b0_idx[mask]
        b1_idx = b1_idx[mask]
        if return_iou:
            return b0_idx, b1_idx, iou_scores[mask]
        else:
            return b0_idx, b1_idx

    def _predict_by_iou_and_assignments(self, boxes_0, boxes_1, assign_0, assign_1, return_iou=False, iou_threshold=None):
        iou_threshold = iou_threshold or self.iou_threshold
        iou_matrix = self._get_iou_matrix(boxes_0, boxes_1)
        b0_idx, b1_idx = linear_sum_assignment(iou_matrix, maximize=True)
        iou_scores = iou_matrix[b0_idx, b1_idx]
        mask = iou_scores > iou_threshold
        b0_idx = b0_idx[mask]
        b1_idx = b1_idx[mask]
        if return_iou:
            return b0_idx, b1_idx, iou_scores[mask]
        else:
            return b0_idx, b1_idx


class Tracker_2(Tracker):
    """
    estimate location 
    """
    def __init__(self, iou_threshold=0.3):
        super().__init__(iou_threshold)
        self.track_locations = {}
        self.track_errors = {}

    def get_location(self, track_ids, box_ids):
        locations = []
        box_valid_ids = []
        box_invalid_ids = []
        for t_id, b_id in zip(track_ids,box_ids):
            location_hist = self.track_locations[t_id]
            error_hist = self.track_errors[t_id]
            num_hist = np.minimum(len(location_hist), 10)
            location_hist = np.array(location_hist[-num_hist:])
            error_hist = np.array(error_hist[-num_hist:])
            location_hist = location_hist[error_hist < -2.]
            location_stability = np.max(np.std(location_hist, axis=0))
            if len(location_hist)>2 and location_stability<0.1:#temporary linear weight
                weight = np.arange(1,len(location_hist)+1).reshape(-1,1)
                location = np.sum(location_hist * weight, axis=0) / np.sum(weight)
                #主成分フィッティングで次の点を予測する
                
                
                #location = np.mean(location_hist, axis=0)
                locations.append(location)
                box_valid_ids.append(b_id)
            else:
                box_invalid_ids.append(b_id)
        return np.array(locations), np.array(box_valid_ids), np.array(box_invalid_ids)
        
    def update_location(self, track_ids, new_locations, icp_errors):
        for t_id, loc, error in zip(track_ids, new_locations, icp_errors):
            if t_id in self.track_locations.keys():                
                self.track_locations[t_id].append(loc)
                self.track_errors[t_id].append(error)
            else:
                self.track_locations[t_id] = [loc]
                self.track_errors[t_id] = [error]
    
    def precheck_iou(self, game_play, view, frame, current_boxes, track_length_thresh=2, iou_threshold=None):
        c_idx = None
        ious = None
        low_iou_c_idx = None
        if len(current_boxes)>0:
            if self.hist[0]["boxes"] is not None:     
                p_idx, c_idx, ious = self._predict_by_iou(self.hist[0]["boxes"], current_boxes, return_iou=True, iou_threshold=iou_threshold)
                #low_iou_c_idx = np.array(list(set(range(len(current_boxes))) - set(c_idx)))
                tracked_id = self.get_tracking_box_id(game_play, view, self.hist[0]["frame"], p_idx)
                tracked_mask = (np.array([len(self.info[t_id]) for t_id in tracked_id])>track_length_thresh)
                c_idx = c_idx[tracked_mask]
                ious = ious[tracked_mask]
                #print(len(current_boxes), len(tracked_mask), tracked_mask.sum())
        return c_idx, ious
    
    def precheck_and_get_location(self, game_play, view, frame, current_boxes):
        #self.set_box_history(game_play, view, frame, current_boxes, conf=conf)
        num_tracked = 0
        c_idx = None
        locations = None
        notrack_c_idx = None
        if len(current_boxes)>0:
            new_track_id = self.current_id + np.arange(len(current_boxes))#dafault
            if self.hist[0]["boxes"] is not None:     
                p_idx, c_idx = self._predict_by_iou(self.hist[0]["boxes"], current_boxes)
                low_iou_c_idx = np.array(list(set(range(len(current_boxes))) - set(c_idx)))
                tracked_id = self.get_tracking_box_id(game_play, view, self.hist[0]["frame"], p_idx)
                locations, c_idx, notrack_c_idx = self.get_location(tracked_id, c_idx)
                notrack_c_idx = np.concatenate([notrack_c_idx, low_iou_c_idx], axis=0)
                if len(c_idx) < 2:
                    c_idx = None
                    locations = None
                    notrack_c_idx = None
        return c_idx, notrack_c_idx, locations
                
    def predict_and_add(self, game_play, view, frame, assigned_player, current_boxes, locations, icp_errors, weight=1.0, conf=None):
        self.set_box_history(game_play, view, frame, current_boxes, locations, conf=conf)
        if len(current_boxes)>0:
            new_track_id = self.current_id + np.arange(len(current_boxes))#dafault
            if self.previous_boxes is not None:     
                prev_boxes = self.estimate_current_frame_box(self.previous_boxes, self.previous_tracked_id)
                p_idx, c_idx = self._predict_by_iou(prev_boxes, current_boxes)
                tracked_id = self.get_tracking_box_id(game_play, view, self.previous_frame, p_idx)
                new_track_id[c_idx] = tracked_id
                not_assigned = [i for i in range(len(current_boxes)) if i not in c_idx]
                num_new_track = len(current_boxes) - len(c_idx)
                new_ids = self.current_id + np.arange(num_new_track)
                new_track_id[not_assigned] = new_ids
                
                p_track = self.previous_boxes[p_idx]
                c_track = current_boxes[c_idx]

            else:
                p_track = None
                c_track = None
                tracked_id = []
                num_new_track = len(current_boxes)
                new_ids = self.current_id + np.arange(num_new_track)
            
            if num_new_track>0 and len(self.pprevious_tracked_id)>0:
                not_assigned_boxes = current_boxes[not_assigned]
                candidate_pprevious_boxes = []
                candidate_track_ids = []
                
                for i, tr_id in enumerate(self.pprevious_tracked_id):
                    if tr_id in self.previous_tracked_id:
                        continue
                    candidate_track_ids.append(tr_id)
                    candidate_pprevious_boxes.append(self.pprevious_boxes[i])
                candidate_pprevious_boxes = np.array(candidate_pprevious_boxes)
                if len(candidate_track_ids)>0:
                    p_idx, c_idx = self._predict_by_iou(candidate_pprevious_boxes, not_assigned_boxes)
                    lost_boxes = candidate_pprevious_boxes[p_idx]
                    emerge_boxes = not_assigned_boxes[c_idx]
                    average_boxes = (lost_boxes + emerge_boxes)/2
                    re_assigned_id = np.array(candidate_track_ids)[p_idx]
                    ###self.add_box_history_and_id(game_play, view, frame-1, average_boxes, re_assigned_id)
                    
                    #re_assign = np.array(not_assigned)[c_idx]
                    #new_track_id[re_assign] = re_assigned_id                
            
            self.add_tracking_speed(p_track, c_track, tracked_id, new_ids)
            self.add_tracking_box_id(game_play, view, frame, new_track_id, assigned_player, weight)
            self.update_location(new_track_id, locations, icp_errors)
            self.pprevious_boxes = self.previous_boxes
            self.pprevious_frame = self.previous_frame
            self.pprevious_tracked_id = self.previous_tracked_id
            self.previous_boxes = current_boxes
            self.previous_frame = frame
            self.previous_tracked_id = new_track_id
            self.current_id += num_new_track
            
        else:
            self.previous_boxes = None
            self.previous_frame = None
            self.previous_tracked_id = []          
            

class Tracker_2_w_feature(Tracker_2):
    """
    estimate location 
    """
    def __init__(self, iou_threshold=0.3):
        super().__init__(iou_threshold)
        self.box_to_feature ={}
        self.feature_bank = {}
        self.max_retracking_interval = 6
        self.hist = {i: {"assignment": None, 
                             "boxes": None,
                             "frame": None,
                             "track_id": [],
                             "feature": None,
                             "lost_track_ids":[],
                             "lost_track_boxes":[]
                             } for i in range(self.max_retracking_interval+1)}
    
    def add_tracking_box_feature(self, game_play, view, frame, feature):
        self.box_to_feature["{}_{}_{}".format(game_play, view, frame)] = feature
     
    def predict_and_add(self, game_play, view, frame, assigned_player, current_boxes, locations, icp_errors, player_feature, weight=1.0, conf=None):
        self.set_box_history(game_play, view, frame, current_boxes, locations, conf=conf, icp_errors=icp_errors)
        if len(current_boxes)>0:
            new_track_id = self.current_id + np.arange(len(current_boxes))#dafault
            new_ious = np.ones((len(current_boxes)))
            if self.hist[0]["boxes"] is not None:
                prev_boxes = self.estimate_current_frame_box(self.hist[0]["boxes"], self.hist[0]["track_id"])

                #p_idx, c_idx, ious = self._predict_by_iou_and_featuredist(prev_boxes, current_boxes, self.hist[0]["feature"], player_feature, return_iou=True)
                p_idx, c_idx, ious = self._predict_by_iou_and_featuredist_assignments(prev_boxes, current_boxes, self.hist[0]["feature"], player_feature, self.hist[0]["assignment"], assigned_player, return_iou=True)
                tracked_id = self.get_tracking_box_id(game_play, view, self.hist[0]["frame"], p_idx)
                
                new_track_id[c_idx] = tracked_id
                new_ious[c_idx] = ious
                not_assigned = [i for i in range(len(current_boxes)) if i not in c_idx]
                num_new_track = len(current_boxes) - len(c_idx)
                new_ids = self.current_id + np.arange(num_new_track)
                new_track_id[not_assigned] = new_ids
                
                p_track = self.hist[0]["boxes"][p_idx]
                c_track = current_boxes[c_idx]

            else:
                p_track = None
                c_track = None
                tracked_id = []
                num_new_track = len(current_boxes)
                new_ids = self.current_id + np.arange(num_new_track)
                not_assigned = [i for i in range(num_new_track)]
            
            
            # retrack more than 2 frames
            if num_new_track > 0:
                not_assigned_boxes = current_boxes[not_assigned]
                not_assigned_player = assigned_player[not_assigned]
                for i in range(self.max_retracking_interval):
                    if len(not_assigned_boxes)>0 and len(self.hist[i]["lost_track_ids"])>0:
                        candidate_boxes = np.array(self.hist[i]["lost_track_boxes"])
                        candidate_track_ids = self.hist[i]["lost_track_ids"]
                            
                        if len(candidate_track_ids)>0:
                            p_idx, c_idx, ious = self._predict_by_iou(candidate_boxes, not_assigned_boxes, return_iou=True)
                            lost_boxes = candidate_boxes[p_idx]
                            emerge_boxes = not_assigned_boxes[c_idx]
                            average_boxes = (lost_boxes + emerge_boxes)/2
                            re_assigned_id = np.array(candidate_track_ids)[p_idx]
                            re_assign = np.array(not_assigned)[c_idx]
                            
                            for j in range(i+1):
                                self.add_box_history_and_id(game_play, view, frame-(j+1), average_boxes, re_assigned_id, ious=ious, conf=0.1)
                            
                            #reuse lost tracking id if iou between current and lost frame
                            new_track_id[re_assign] = re_assigned_id                
                            new_ious[re_assign] = ious 
                            
                            still_lost_idx = set(range(len(candidate_track_ids))) - set(p_idx)
                            self.hist[i]["lost_track_ids"] = [self.hist[i]["lost_track_ids"][idx] for idx in still_lost_idx]
                            self.hist[i]["lost_track_boxes"] = [self.hist[i]["lost_track_boxes"][idx] for idx in still_lost_idx]
                            #"""
                            
                            still_not_assigned = np.array(list(set(range(len(not_assigned_boxes))) - set(c_idx)))
                            if len(still_not_assigned)>0:
                                #print(still_not_assigned, not_assigned_boxes, not_assigned_player)
                                not_assigned_boxes = not_assigned_boxes[still_not_assigned]
                                not_assigned_player = not_assigned_player[still_not_assigned]
                            else:
                                break
                        
                        
            
            self.add_tracking_speed(p_track, c_track, tracked_id, new_ids)

            self.add_tracking_box_id(game_play, view, frame, new_track_id, assigned_player, weight, ious=new_ious)
            self.add_tracking_box_feature(game_play, view, frame, player_feature)
            self.update_location(new_track_id, locations, icp_errors)
            self.current_id += num_new_track
            for i in range(self.max_retracking_interval)[::-1]:
                self.hist[i+1] = self.hist[i].copy()
            
            lost_track_boxes = []
            lost_track_ids = []
                
            for i, tr_id in enumerate(self.hist[0]["track_id"]):
                if tr_id in new_track_id:
                    continue
                lost_track_ids.append(tr_id)
                lost_track_boxes.append(self.hist[0]["boxes"][i])
                    
            self.hist[0] = {"assignment": assigned_player, 
                             "boxes": current_boxes,
                             "frame": frame,
                             "track_id": new_track_id,
                             "feature": player_feature,
                             "lost_track_ids": lost_track_ids,
                             "lost_track_boxes": lost_track_boxes,
                             }
        else:
            self.previous_assignment = None
            self.previous_boxes = None
            self.previous_frame = None
            self.previous_tracked_id = []            
            
        
    def _predict_by_iou_and_featuredist(self, boxes_0, boxes_1, b0_feature, b1_feature, feature_weight=0.2, return_iou=False):
        iou_matrix = self._get_iou_matrix(boxes_0, boxes_1)
        dist_cost = -feature_weight * np.sum((b0_feature[:,np.newaxis,:] - b1_feature[np.newaxis,:,:])**2, axis=-1)
        dist_cost = np.where(iou_matrix>self.iou_threshold, dist_cost, -4.0)#kasanari
        b0_idx, b1_idx = linear_sum_assignment(iou_matrix+dist_cost*1.0, maximize=True)
        iou_scores = iou_matrix[b0_idx, b1_idx]
        mask = iou_scores > self.iou_threshold
        b0_idx = b0_idx[mask]
        b1_idx = b1_idx[mask]
        if return_iou:
            return b0_idx, b1_idx, iou_scores[mask]
        else:
            return b0_idx, b1_idx        

    def _predict_by_iou_and_featuredist_assignments(self, boxes_0, boxes_1, b0_feature, b1_feature, b0_assign, b1_assign, feature_weight=0.2, assign_weight=0.2, return_iou=False):
        iou_matrix = self._get_iou_matrix(boxes_0, boxes_1)
        dist_cost = -feature_weight * np.sum((b0_feature[:,np.newaxis,:] - b1_feature[np.newaxis,:,:])**2, axis=-1)
        dist_cost = np.where(iou_matrix>self.iou_threshold, dist_cost, -4.0)#kasanari
        assign_cost = assign_weight * (np.array(b0_assign)[:,np.newaxis] == np.array(b1_assign)[np.newaxis:]).astype(float)
        assign_cost = np.where(iou_matrix>self.iou_threshold, assign_cost, 0.0)#kasanari
        b0_idx, b1_idx = linear_sum_assignment(iou_matrix+dist_cost+assign_cost, maximize=True)
        iou_scores = iou_matrix[b0_idx, b1_idx]
        mask = iou_scores > self.iou_threshold
        b0_idx = b0_idx[mask]
        b1_idx = b1_idx[mask]
        if return_iou:
            return b0_idx, b1_idx, iou_scores[mask]
        else:
            return b0_idx, b1_idx 
    
def ensemble_reassign_player_label(trackers, game_play, view, frame, frame_sigma=5.0):
    all_players_score = {}
    track_ids = [trk.box_to_id["{}_{}_{}".format(game_play, view, frame)] for trk in trackers]
    num_box = len(track_ids[0])    
    
    for box_idx, track_id_each_trk in enumerate(zip(*track_ids)):
        players_score = {}
        for track_id, trk in zip(track_id_each_trk, trackers):
            track_info = trk.info[track_id]
            track_weights = trk.weights[track_id]
            for track_frame, player in track_info.items():
                weight = track_weights[track_frame]
                dev_frame = np.abs(track_frame - frame)
                score = weight * np.exp(-dev_frame/frame_sigma)
                if player in players_score.keys():
                    players_score[player] += score
                else:
                    players_score[player] = score
        for p, s in players_score.items():
            if p in all_players_score.keys():
                all_players_score[p][box_idx] = s
            else:
                scores_list = np.zeros((num_box))
                scores_list[box_idx] = s
                all_players_score[p] = scores_list            

    
    cost_matrix = np.array(list(all_players_score.values())).T
    player_labels = list(all_players_score.keys())
    if cost_matrix.shape[0]>cost_matrix.shape[1]:
        raise Exception("tracked player is fewer than boxes. why?")
    _, players_idx = linear_sum_assignment(cost_matrix, maximize=True)
    new_assignments = [player_labels[idx] for idx in players_idx] 
    return new_assignments

def get_ious(boxes, box):
    """
    boxes [N0, 4(top left bottom right)], 
    box [1, 4(top left bottom right)]
    return:
        iou_matrix [N0, N1]
    """

    tops = np.maximum(boxes[:,0], box[:,0])
    lefts = np.maximum(boxes[:,1], box[:,1])
    bottoms = np.minimum(boxes[:,2], box[:,2])
    rights = np.minimum(boxes[:,3], box[:,3])
    intersection = np.maximum(bottoms-tops,0) * np.maximum(rights-lefts,0)
    area_0 = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    area_1 = (box[:,2] - box[:,0]) * (box[:,3] - box[:,1])
    union = area_0 + area_1 - intersection
    ious = intersection/(union+1e-7)
    return ious



def wbf(list_boxes, list_scores, list_others, model_weights=None, iou_thresh=0.8, mode="average"):
    """
    inputs are list of numpy array
    assuimng single class
    """
    if model_weights is None:
        model_weights = np.ones(len(list_boxes))
    if mode=="average":
        model_weights = np.array(model_weights) / np.sum(model_weights)
    elif mode=="maximum":
        model_weights = np.array(model_weights) / np.max(model_weights)
    else:
        raise Exception()
        
    # set initial detecions
    fusion_boxes = list_boxes[0]
    fusion_confidence = list_scores[0] * model_weights[0]
    fusion_others = list_others[0] * model_weights[0]
    #fusion_counts = np.array([model_weights[0]]*len(list_boxes[0]))
    for boxes, scores, others, weight in zip(list_boxes[1:], list_scores[1:], list_others[1:], model_weights[1:]):
        for box, score, other in zip(boxes, scores, others):
            ious = get_ious(fusion_boxes, box.reshape(1,4))
            argmax = np.argmax(ious)
            max_iou = ious[argmax]
            if max_iou > iou_thresh:
                f_box = fusion_boxes[argmax]
                f_conf = fusion_confidence[argmax]
                f_other = fusion_others[argmax]
                add_box = box
                add_score = score * weight
                add_other = other * weight
                if mode=="average":
                    new_conf = f_conf + add_score
                    new_box = ((f_box * f_conf) + (add_box * add_score)) / new_conf
                    new_other = f_other + add_other
                elif mode=="maximum":
                    new_conf = max(f_conf, add_score)
                    new_box = ((f_box * f_conf) + (add_box * add_score)) / (f_conf + add_score)
                    new_other = f_other if f_conf>add_score else add_other
                fusion_boxes[argmax] = new_box
                fusion_confidence[argmax] = new_conf
                fusion_others[argmax] = new_other
            else:
                fusion_boxes = np.concatenate([fusion_boxes, box.reshape(1,4)], axis=0)
                fusion_confidence = np.append(fusion_confidence, score * weight)
                fusion_others = np.concatenate([fusion_others, other.reshape(1,-1) * weight], axis=0)
    return fusion_boxes, fusion_confidence, fusion_others

def get_iou_matrix(boxes_0, boxes_1):
    """
    boxes [N0, 4(top left bottom right)], [N1, 4(top left bottom right)]
    #left", "width", "top", "height"
    return:
        iou_matrix [N0, N1]
    """
    boxes_0 = boxes_0[:,np.newaxis,:]
    boxes_1 = boxes_1[np.newaxis,:,:]
    tops = np.maximum(boxes_0[:,:,0], boxes_1[:,:,0])
    lefts = np.maximum(boxes_0[:,:,1], boxes_1[:,:,1])
    bottoms = np.minimum(boxes_0[:,:,2], boxes_1[:,:,2])
    rights = np.minimum(boxes_0[:,:,3], boxes_1[:,:,3])
    intersection = np.maximum(bottoms-tops,0) * np.maximum(rights-lefts,0)
    area_0 = (boxes_0[:,:,2] - boxes_0[:,:,0]) * (boxes_0[:,:,3] - boxes_0[:,:,1])
    area_1 = (boxes_1[:,:,2] - boxes_1[:,:,0]) * (boxes_1[:,:,3] - boxes_1[:,:,1])
    union = area_0 + area_1 - intersection
    iou_matrix = intersection/(union+1e-7)
    return iou_matrix

def wbf_2(list_boxes, list_scores, list_others, model_weights=None, iou_thresh=0.8, mode="average"):
    """
    inputs are list of numpy array
    assuimng single class
    """
    if model_weights is None:
        model_weights = np.ones(len(list_boxes))
    if mode=="average":
        model_weights = np.array(model_weights) / np.sum(model_weights)
    elif mode=="maximum":
        model_weights = np.array(model_weights) / np.max(model_weights)
    else:
        raise Exception()
    # set initial detecions
    fusion_boxes = list_boxes[0]
    fusion_confidence = list_scores[0] * model_weights[0]
    fusion_others = list_others[0] * model_weights[0]
    #fusion_counts = np.array([model_weights[0]]*len(list_boxes[0]))
    for boxes, scores, others, weight in zip(list_boxes[1:], list_scores[1:], list_others[1:], model_weights[1:]):
        iou_matrix = get_iou_matrix(fusion_boxes, boxes)#[Nf,N]
        num_1, num_2 = iou_matrix.shape
        b1_idx = set(range(num_1))
        b2_idx = set(range(num_2))
        while True:
            if len(b1_idx)==0 or len(b2_idx)==0:
                break
            argmax = np.argmax(iou_matrix)
            row, col = argmax//num_2, argmax%num_2            
            max_iou = iou_matrix[row, col]
            
            if max_iou > iou_thresh:
                f_box = fusion_boxes[row]
                f_conf = fusion_confidence[row]
                f_other = fusion_others[row]
                add_box = boxes[col]
                add_score = scores[col] * weight
                add_other = others[col] * weight
                if mode=="average":
                    new_conf = f_conf + add_score
                    new_box = ((f_box * f_conf) + (add_box * add_score)) / new_conf
                    new_other = f_other + add_other
                elif mode=="maximum":
                    new_conf = max(f_conf, add_score)
                    new_box = ((f_box * f_conf) + (add_box * add_score)) / (f_conf + add_score)
                    new_other = f_other if f_conf>add_score else add_other
                fusion_boxes[row] = new_box
                fusion_confidence[row] = new_conf
                fusion_others[row] = new_other
            else:
                fusion_boxes = np.concatenate([fusion_boxes, boxes[col].reshape(1,4)], axis=0)
                fusion_confidence = np.append(fusion_confidence, scores[col] * weight)
                fusion_others = np.concatenate([fusion_others, others[col].reshape(1,-1) * weight], axis=0)
        
            iou_matrix[:,col] = -1.
            b1_idx = b1_idx - set([row])
            b2_idx = b2_idx - set([col])
                
    return fusion_boxes, fusion_confidence, fusion_others

def wbf_hangarian(list_boxes, list_scores, list_others, model_weights=None, iou_thresh=0.8, mode="average"):
    """
    inputs are list of numpy array
    assuimng single class
    """
    if model_weights is None:
        model_weights = np.ones(len(list_boxes))
    if mode=="average":
        model_weights = np.array(model_weights) / np.sum(model_weights)
    elif mode=="maximum":
        model_weights = np.array(model_weights) / np.max(model_weights)
    else:
        raise Exception()
        
    # set initial detecions
    fusion_boxes = list_boxes[0]
    fusion_confidence = list_scores[0] * model_weights[0]
    fusion_others = list_others[0] * model_weights[0]
    #fusion_counts = np.array([model_weights[0]]*len(list_boxes[0]))
    for boxes, scores, others, weight in zip(list_boxes[1:], list_scores[1:], list_others[1:], model_weights[1:]):
        iou_matrix = get_iou_matrix(fusion_boxes, boxes)#[Nf,N]
        num_b0, num_b1 = iou_matrix.shape
        b0_idx, b1_idx = linear_sum_assignment(iou_matrix, maximize=True)
        ious = iou_matrix[b0_idx, b1_idx]
        boxes_assign = boxes[b1_idx]
        scores_assign = scores[b1_idx]
        others_assign = others[b1_idx]
        
        not_assigned_b1_idx = np.array(list(set(range(num_b1)) - set(b1_idx))).astype(int)
        
        boxes_not_assign = boxes[not_assigned_b1_idx]
        scores_not_assign = scores[not_assigned_b1_idx]
        others_not_assign = others[not_assigned_b1_idx]
        
        
        for box, score, other, max_iou, argmax in zip(boxes_assign, scores_assign, others_assign, ious, b0_idx):            
            #argmax = np.argmax(ious)
            #max_iou = ious[argmax]
            if max_iou > iou_thresh:
                f_box = fusion_boxes[argmax]
                f_conf = fusion_confidence[argmax]
                f_other = fusion_others[argmax]
                add_box = box
                add_score = score * weight
                add_other = other * weight
                if mode=="average":
                    new_conf = f_conf + add_score
                    new_box = ((f_box * f_conf) + (add_box * add_score)) / new_conf
                    new_other = f_other + add_other
                elif mode=="maximum":
                    new_conf = max(f_conf, add_score)
                    new_box = ((f_box * f_conf) + (add_box * add_score)) / (f_conf + add_score)
                    new_other = f_other if f_conf>add_score else add_other
                fusion_boxes[argmax] = new_box
                fusion_confidence[argmax] = new_conf
                fusion_others[argmax] = new_other
            else:
                fusion_boxes = np.concatenate([fusion_boxes, box.reshape(1,4)], axis=0)
                fusion_confidence = np.append(fusion_confidence, score * weight)
                fusion_others = np.concatenate([fusion_others, other.reshape(1,-1) * weight], axis=0)
        
        for box, score, other in zip(boxes_not_assign, scores_not_assign, others_not_assign):            
            fusion_boxes = np.concatenate([fusion_boxes, box.reshape(1,4)], axis=0)
            fusion_confidence = np.append(fusion_confidence, score * weight)
            fusion_others = np.concatenate([fusion_others, other.reshape(1,-1) * weight], axis=0)
    return fusion_boxes, fusion_confidence, fusion_others



def wbf_ensemble_reassign_player_label(trackers, game_play, view, frame, frame_sigma=5.0,
                                       model_weights = None, wbf_iou=0.45):
    
    track_ids_all = [trk.box_to_id["{}_{}_{}".format(game_play, view, frame)] for trk in trackers]
    
    #num_box = len(track_ids[0])    
    list_fusion_boxes = []
    list_fusion_confidence = []
    list_player_scores_dict = []
    
    all_player_labels = []
    nums_box = []

    
    for i, [track_ids, trk] in enumerate(zip(track_ids_all, trackers)):
        all_players_score = {}
        num_box = len(track_ids)
        nums_box.append(num_box)
        for box_idx, track_id in enumerate(track_ids):
            track_info = trk.info[track_id]
            track_weights = trk.weights[track_id]
            track_iou_weights = trk.iou_weights[track_id]
                        
            players_score = {}
            frame_list = list(track_info.keys())
            frame_idx = frame_list.index(frame)
            #weights = np.array([track_weights[track_frame] for track_frame in frame_list])
            weights = np.array(list(track_weights.values()))
            
            # iou-based decay is better than simple decay(frame distance)
            iou_weights = np.array(list(track_iou_weights.values()))
            iou_weights = np.cumsum(iou_weights)
            iou_weights = np.abs(iou_weights - iou_weights[frame_idx])
            decay = 2.5**(-iou_weights)*2
            
            #dev_frames = np.abs(np.array(frame_list)-frame)
            #simple_decay = np.exp(-dev_frames/frame_sigma)
            #decay=simple_decay
            scores = weights * decay#/np.sqrt(decay.sum())
            
            for score, player in zip(scores, track_info.values()):
                if player == "invalid_player":
                    continue
                if player in players_score.keys():
                    players_score[player] += score
                else:
                    players_score[player] = score
                    if not player in all_player_labels:
                        all_player_labels.append(player)
            
            for p, s in players_score.items():
                if p in all_players_score.keys():
                    all_players_score[p][box_idx] = s
                else:
                    scores_list = np.zeros((num_box))
                    scores_list[box_idx] = s
                    all_players_score[p] = scores_list

        list_fusion_boxes.append(trk.box_history["{}_{}_{}".format(game_play, view, frame)])
        list_fusion_confidence.append(trk.box_history_conf["{}_{}_{}".format(game_play, view, frame)])
        list_player_scores_dict.append(all_players_score)
    
    list_fusion_player_scores = []
    for num_box, player_score in zip(nums_box, list_player_scores_dict):
        for player in all_player_labels:
            if not player in player_score.keys():
                player_score[player] = np.zeros((num_box))
        list_scores = [player_score[p] for p in all_player_labels]
        list_fusion_player_scores.append(np.array(list_scores).T)
    
    f_boxes, f_confs, f_p_scores = wbf_hangarian(list_fusion_boxes, 
                                       list_fusion_confidence, 
                                       list_fusion_player_scores, 
                                       model_weights=model_weights, 
                                       iou_thresh=wbf_iou, 
                                       mode="average",
                                       )
    
    cost_matrix = f_p_scores
    
    pred_idx, players_idx = linear_sum_assignment(cost_matrix, maximize=True)
    new_assignments = [all_player_labels[idx] for idx in players_idx]
    costs = cost_matrix[pred_idx, players_idx]
    nonzero_mask = (costs!=0)
    f_boxes = f_boxes[pred_idx]
    f_confs = f_confs[pred_idx]
    return f_boxes[nonzero_mask], f_confs[nonzero_mask], np.array(new_assignments)[nonzero_mask]
    