# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:18:25 2021

@author: ras
"""
import numpy as np

def similarity_matrix_to_team(simmat, threshold=0.5):
    #similarity matrix [N,N] -> [N] binary
    
    num_predicts = len(simmat)
    index_list = np.arange(num_predicts)

    #まず真ん中を消す
    simmat = simmat * (1.0-np.eye(num_predicts))
    original_simmat = simmat.copy()
    
    team_a_idx = []
    #team_b_idx = []

    #choose similar samples from top
    argmax = np.argmax(simmat)
    row, column = argmax%num_predicts, argmax//num_predicts
    team_a_idx.append(index_list[row])
    index_list = np.delete(index_list, row)
    simmat = np.delete(simmat, row, axis=1)
    while simmat.shape[1]>0:
        sim_mean = simmat[team_a_idx].mean(axis=0)
        argmax = np.argmax(sim_mean)
        if sim_mean[argmax] < threshold:
            break
        team_a_idx.append(index_list[argmax])
        simmat = np.delete(simmat, argmax, axis=1)
        index_list = np.delete(index_list, argmax)
    team_b_idx = index_list.tolist()#残りもの
    
    #scoring after team grouping
    team_a_score_pos = np.sum(original_simmat[team_a_idx, :][:, team_a_idx], axis=-1)#/(len(team_a_idx)-1)
    team_b_score_pos = np.sum(original_simmat[team_b_idx, :][:, team_b_idx], axis=-1)#/(len(team_b_idx)-1)
    team_a_score_neg = np.sum(1.0 - original_simmat[team_a_idx, :][:, team_b_idx], axis=-1)
    team_b_score_neg = np.sum(1.0 - original_simmat[team_b_idx, :][:, team_a_idx], axis=-1)
    
    team_a_score = (team_a_score_pos + team_a_score_neg)/(num_predicts-1)
    team_b_score = (1.0 - (team_b_score_pos+ team_b_score_neg)/(num_predicts-1))
    
    scores = np.zeros((num_predicts))
    scores[team_a_idx] = team_a_score
    scores[team_b_idx] = team_b_score
    
    return scores.astype(np.float32)
    

if __name__ == "__main__":
    simmat = np.array([[1.0, 0.9, 0.3, 0.2, 0.8],
                       [0.9, 1.0, 0.1, 0.3, 0.6],
                       [0.3, 0.1, 1.0, 0.78, 0.25],
                       [0.2, 0.3, 0.78, 1.0, 0.26],
                       [0.8, 0.6, 0.25, 0.26, 1.0]])
    print(similarity_matrix_to_team(simmat, threshold=0.5))