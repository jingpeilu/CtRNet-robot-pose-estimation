import os
import sys
import numpy as np
from utils import *
from models.BPnP import batch_transform_3d

def evaluate_ADD(CtRNet, dataset, use_gpu=True):
    from tqdm.notebook import tqdm

    err_3d_list = list()
    err_2d_list = list()

    for data_n in tqdm(range(len(dataset))):
        img, joint_angles, keypoints = dataset.get_data_with_keypoints(data_n) 

        if use_gpu:        
            img = img.cuda()

        cTr, points_2d, segmentation, heatmap = CtRNet.inference_single_image(img, joint_angles.cpu().squeeze())

        # get ground truth
        points_2d_gt = list()
        points_3d_gt = list()
        for i in range(len(keypoints)):
            points_2d_gt.append(keypoints[i]['projected_location'])
            points_3d_gt.append(keypoints[i]['location'])

        points_2d_gt = np.array(points_2d_gt)
        points_3d_gt = np.array(points_3d_gt)


        
        # compute 3d keypoints
        _,t_list = CtRNet.robot.get_joint_RT(joint_angles.cpu().squeeze())
        points_3d = torch.from_numpy(t_list).float().cuda()

        points_3d_pred = to_np(batch_transform_3d(cTr, points_3d)).squeeze()
        points_3d_pred = points_3d_pred[[0,2,3,4,6,7,8]]



        if points_3d_pred[0,-1] < 0:
            points_3d_pred = -points_3d_pred

        err_3d = np.linalg.norm(points_3d_pred - points_3d_gt,axis=1)
        err_3d = np.mean(err_3d)
        err_3d_list.append(err_3d)

    err_3d_list = np.array(err_3d_list).flatten()
    err_3d_list.sort()

    ADD = list()
    for i in range(1000):
        num = np.sum(err_3d_list < i/10000.0) / err_3d_list.shape[0]
        ADD.append(num)

    np.sum(ADD)/len(ADD)

    return np.sum(ADD)/len(ADD), np.mean(err_3d_list) # AUC and mean error