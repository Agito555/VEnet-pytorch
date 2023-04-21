# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds) # (B , num_seed), to indicate which the point belong to, backgrond or foreground
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR) # to get the inds
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3) # the gt vote from seed

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3 
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    # dist2: (B * num_seed, GT_VOTE_FACTOR), GT_VOTE_FACTOR=3
    dist1, idx1, dist2, idx2 = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)

    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss


def compute_vote_attraction_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_attraction_loss: scalar Tensor
            
    Overall idea:
        
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]
    instance_label=end_points['instance_label']

    # seed_inds: bs * num_seed 
    # instance_label: bs * num_point 
    seed_instance_label=torch.zeros_like(seed_inds) # to indicate which instance label the points belong to
    for bs in range(batch_size):
        seed_instance_label[bs,:]=instance_label[bs,seed_inds[bs,:]]

    seed_instance_label=seed_instance_label.cpu()
        
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds).cpu() # (B , num_seed)
    
    vote_attraction_loss_tmp=0
    num_instance=0
    vote_attraction_loss=0

    for bs in range(batch_size):
        labels=np.unique(seed_instance_label[bs,:])
        num_instance_tmp=len(labels)-1 if 0 in labels else len(labels)
        num_instance+=num_instance_tmp

        for label in labels:
            if label==0:
                continue
            else:
                idicator=seed_instance_label[bs,:]==label
                vote_tmp=vote_xyz[bs,idicator,:]
                num_vote=vote_tmp.shape[0]
                vote_center=torch.mean(vote_tmp,dim=0)

                # l1 loss
                vote_attraction_loss_tmp+=torch.dist(vote_tmp,vote_center,p=1)/num_vote
                

    #     # if average per batch
    #     vote_attraction_loss+=vote_attraction_loss_tmp/num_instance_tmp
    #     vote_attraction_loss_tmp=0

    # # if average per batch
    # vote_attraction_loss/=batch_size

    # if average in all batch
    vote_attraction_loss = vote_attraction_loss_tmp/num_instance

    return vote_attraction_loss


def compute_seed_cls_loss(end_points,pos_weight=torch.tensor([4]),weight=torch.tensor([0.25])):
    """ Compute seed cls loss.

    Args:
        end_points: dict (read-only)
    
    Returns:
        seed_cls_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to predict to be object.
        Otherwise,inversly.

    """

    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds).float() # B,num_seed
    # seed_gt_votes_mask=seed_gt_votes_mask.reshape(-1)

    #for debug use
    # seed_gt_votes_mask=end_points['seed_gt_votes_mask']
    # print(type(seed_gt_votes_mask))

    prob_logits=end_points['prob_logits']
    # prob=prob.reshape(-1)

    # maybe there are problems during multiple gpu training
    cls_loss=F.binary_cross_entropy_with_logits(prob_logits,seed_gt_votes_mask,reduction='mean',pos_weight=pos_weight.to("cuda:0"),\
                                                weight=weight.to("cuda:0"))
    # cls_loss=F.binary_cross_entropy_with_logits(prob_logits,seed_gt_votes_mask,reduction='none',pos_weight=pos_weight,\
    #                                             weight=weight)

    return cls_loss


def compute_seed_focal_loss(end_points,gamma=2,pos_weight=1/3,weight=0.75):

    # alpha=0.25 in default,so pos=1:neg=3
    
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds).float() # B,num_seed

    # for debug use
    # seed_gt_votes_mask=end_points['seed_gt_votes_mask']

    prob_logits=end_points['prob_logits']
    
    BCE_loss = F.binary_cross_entropy_with_logits(prob_logits,seed_gt_votes_mask,reduction='none')
    pt = torch.exp(-BCE_loss)
    
    weight_tensor=torch.ones_like(BCE_loss)*weight
    weight_tensor[seed_gt_votes_mask==1]*=pos_weight
    weight_tensor=weight_tensor.to("cuda:0")
    
    # BCE_loss=F.binary_cross_entropy_with_logits(prob_logits,seed_gt_votes_mask,reduction='none',pos_weight=pos_weight,\
    #                                             weight=weight)
    

    F_loss =  weight_tensor*(1-pt)**gamma * BCE_loss

    # print('BCE_loss:')
    # print(BCE_loss)
    # print('pt:')
    # print(pt)
    # print('F_loss:')
    # print(F_loss)

    return torch.mean(F_loss)
    
    

def compute_objectness_loss(end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask,prob
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # vote attraction loss
    vote attraction loss
    vote_attraction_loss= compute_vote_attraction_loss(end_points)
    end_points['vote_attraction_loss'] = vote_attraction_loss

    # seed cls loss
    seed_cls_loss=compute_seed_cls_loss(end_points,pos_weight=torch.tensor([8]),weight=torch.tensor([1/8]))
    end_points['seed_cls_loss'] = seed_cls_loss

    # seed focal loss
    # seed_focal_loss=compute_seed_focal_loss(end_points,pos_weight=8,weight=1)
    # end_points['seed_focal_loss'] = seed_focal_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss

    # Final loss function
    # loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss  # original loss
    # loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + seed_cls_loss
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss +0.5* vote_attraction_loss + seed_cls_loss
    # loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss +0.5* vote_attraction_loss + seed_focal_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points


if __name__=='__main__':
    end_points={}
    # end_points['prob']=torch.tensor([[0.2,0.46],[0.1,0.6],[0.7,0.6]])
    # end_points['seed_gt_votes_mask']=torch.tensor([[0.,1.],[0.,1.],[1.,1.]])

    end_points['prob_logits']=torch.tensor([[-1-np.log(1-1/np.e),-1-np.log(1-1/np.e)],[np.log(np.e-1),torch.tensor(np.log(np.e-1))]])
    end_points['seed_gt_votes_mask']=torch.tensor([[0.,0.],[1.,1.]])
    
    # print(compute_seed_cls_loss(end_points,pos_weight=torch.tensor([1]),weight=torch.tensor([1])))
    print(compute_seed_focal_loss(end_points))
    
    