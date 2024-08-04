import os
import time

from PIL import Image

import cv2
import numpy as np
import torch
from progressbar import ProgressBar
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import glob
import pickle
import json

from utils import find_ndds_data_in_dir, transform_DREAM_to_CPLSim_TCR


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def load_camera_parameters(data_folder):
    _, ndds_data_configs = find_ndds_data_in_dir(data_folder)
    with open(ndds_data_configs['camera'], "r") as json_file:
        data = json.load(json_file)

    fx = data['camera_settings'][0]['intrinsic_settings']['fx']
    fy = data['camera_settings'][0]['intrinsic_settings']['fy']
    cx = data['camera_settings'][0]['intrinsic_settings']['cx']
    cy = data['camera_settings'][0]['intrinsic_settings']['cy']

    return fx, fy, cx, cy


class LabelGenerator():
    def __init__(self, args, data_folder):
        self.data_folder = data_folder
        self.ndds_dataset, self.ndds_data_configs = find_ndds_data_in_dir(self.data_folder)
        
        from models.robot_arm import PandaArm
        from models.mesh_renderer import RobotMeshRenderer

        if args.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        intrinsics = np.array([[ args.fx / args.scale     ,    0.   ,   args.px / args.scale ],
                                [  0.        ,  args.fy / args.scale,   args.py / args.scale ],
                                [  0.        ,    0.   ,   1.      ]])

        self.K = torch.tensor(intrinsics, device=self.device, dtype=torch.float)


        urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")
        self.panda_arm = PandaArm(urdf_file=urdf_file)


        focal_length = [-args.fx / args.scale,-args.fy / args.scale]
        principal_point = [args.px / args.scale, args.py / args.scale]
        image_size = [int(args.height / args.scale), int(args.width / args.scale)]
        mesh_files = [args.base_dir + "/urdfs/Panda/meshes/visual/link0/link0.obj",
                    args.base_dir + "/urdfs/Panda/meshes/visual/link1/link1.obj",
                    args.base_dir + "/urdfs/Panda/meshes/visual/link2/link2.obj",
                    args.base_dir + "/urdfs/Panda/meshes/visual/link3/link3.obj",
                    args.base_dir + "/urdfs/Panda/meshes/visual/link4/link4.obj",
                    args.base_dir + "/urdfs/Panda/meshes/visual/link5/link5.obj",
                    args.base_dir + "/urdfs/Panda/meshes/visual/link6/link6.obj",
                    args.base_dir + "/urdfs/Panda/meshes/visual/link7/link7.obj",
                    args.base_dir + "/urdfs/Panda/meshes/visual/hand/hand.obj",
                    ]
        
        self.panda_renderer = RobotMeshRenderer(
            focal_length=focal_length, principal_point=principal_point, image_size=image_size, 
            robot=self.panda_arm, mesh_files=mesh_files, device=self.device)

    def generate_mask(self, idx):

        data_sample = self.ndds_dataset[idx]

        mask_path = data_sample['data_path'].replace('.json', '.mask.npy')
        if os.path.exists(mask_path):
            return mask_path


        label_json = data_sample["data_path"]
        with open(label_json, "r") as json_file:
            data = json.load(json_file)

        joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                data['sim_state']['joints'][1]['position'],
                                data['sim_state']['joints'][2]['position'],
                                data['sim_state']['joints'][3]['position'],
                                data['sim_state']['joints'][4]['position'],
                                data['sim_state']['joints'][5]['position'],
                                data['sim_state']['joints'][6]['position']])


        TCR_ndds = np.array(data['objects'][0]['pose_transform'])
        base_to_cam = transform_DREAM_to_CPLSim_TCR(TCR_ndds)

        base_to_cam = torch.tensor(base_to_cam, dtype=torch.float, device=self.panda_renderer.device)

        robot_mesh = self.panda_renderer.get_robot_mesh(joint_angle)
        rendered_image = self.panda_renderer.silhouette_renderer(meshes_world=robot_mesh, R = base_to_cam[:3,:3].T.unsqueeze(0), T = base_to_cam[:3,3].unsqueeze(0))
        mask = rendered_image[..., 3].cpu().detach().numpy().squeeze()
        np.save(mask_path, mask)

        return mask_path
    
    def generate_keypoints(self, idx):
        from models.BPnP import batch_project

        data_sample = self.ndds_dataset[idx]

        keypoint_path = data_sample['data_path'].replace('.json', '.keypoint.npy')
        if os.path.exists(keypoint_path):
            return keypoint_path


        label_json = data_sample["data_path"]
        with open(label_json, "r") as json_file:
            data = json.load(json_file)

        joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                data['sim_state']['joints'][1]['position'],
                                data['sim_state']['joints'][2]['position'],
                                data['sim_state']['joints'][3]['position'],
                                data['sim_state']['joints'][4]['position'],
                                data['sim_state']['joints'][5]['position'],
                                data['sim_state']['joints'][6]['position']])


        TCR_ndds = np.array(data['objects'][0]['pose_transform'])
        base_to_cam = transform_DREAM_to_CPLSim_TCR(TCR_ndds)
        base_to_cam = torch.tensor(base_to_cam, dtype=torch.float, device=self.panda_renderer.device)

        points = self.panda_arm.get_3d_keypoints(joint_angle)
        points_3d = torch.from_numpy(points).float().to(self.device)
        points_2d_gt = batch_project(base_to_cam[:3,:4][None], points_3d, self.K, angle_axis=False)
        points_2d_gt = points_2d_gt.squeeze().cpu().detach().numpy()
        np.save(keypoint_path, points_2d_gt)

        return keypoint_path
    


class ImageDataLoaderSynthetic(Dataset):

    def __init__(self, data_folder, scale=1, trans_to_tensor=None):
        self.trans_to_tensor = trans_to_tensor
        self.data_folder = data_folder

        self.ndds_dataset, self.ndds_data_configs = find_ndds_data_in_dir(self.data_folder)

        self.scale = scale



    def __len__(self):

        return len(self.ndds_dataset)

    def __getitem__(self, idx):

        data_sample = self.ndds_dataset[idx]
        # load image
        img_path = data_sample["image_paths"]["rgb"]
        image_pil = pil_loader(img_path)
        if self.scale != 1:
            width, height = image_pil.size
            new_size = (int(width*self.scale),int(height*self.scale))
            image_pil = image_pil.resize(new_size)
        image = self.trans_to_tensor(image_pil)

        label_json = data_sample["data_path"]
        with open(label_json, "r") as json_file:
            data = json.load(json_file)

        if "panda" in self.data_folder:
            joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                    data['sim_state']['joints'][1]['position'],
                                    data['sim_state']['joints'][2]['position'],
                                    data['sim_state']['joints'][3]['position'],
                                    data['sim_state']['joints'][4]['position'],
                                    data['sim_state']['joints'][5]['position'],
                                    data['sim_state']['joints'][6]['position']])


            TCR_ndds = np.array(data['objects'][0]['pose_transform'])
            base_to_cam = transform_DREAM_to_CPLSim_TCR(TCR_ndds)

            joint_angle = torch.tensor(joint_angle, dtype=torch.float)
            base_to_cam = torch.tensor(base_to_cam, dtype=torch.float)

        else:
            joint_angle = None
            base_to_cam = None

        # load segmentation mask
        mask_path = data_sample['data_path'].replace('.json', '.mask.npy')
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            if self.scale != 1:
                mask = cv2.resize(mask, new_size)
            mask = torch.tensor(mask, dtype=torch.float)
        else:
            mask = None

        # load keypoints
        keypoint_path = data_sample['data_path'].replace('.json', '.keypoint.npy')
        if os.path.exists(keypoint_path):
            keypoints = np.load(keypoint_path)
            if self.scale != 1:
                keypoints = keypoints * self.scale
            keypoints = torch.tensor(keypoints, dtype=torch.float)
        else:
            keypoints = None

        return image, joint_angle, base_to_cam, keypoints, mask




class ImageDataLoaderReal(Dataset):

    def __init__(self, data_folder, scale=1, trans_to_tensor=None):
        self.trans_to_tensor = trans_to_tensor
        self.data_folder = data_folder

        # if data_folder is a list of folders, then load all the data
        if isinstance(data_folder, list):
            self.ndds_dataset = []
            for folder in data_folder:
                ndds_dataset, _ = find_ndds_data_in_dir(folder)
                self.ndds_dataset.extend(ndds_dataset)
        else:            
            self.ndds_dataset, self.ndds_data_configs = find_ndds_data_in_dir(self.data_folder)

        self.scale = scale



    def __len__(self):

        return len(self.ndds_dataset)

    def __getitem__(self, idx):

        data_sample = self.ndds_dataset[idx]
        # load image
        img_path = data_sample["image_paths"]["rgb"]
        image_pil = pil_loader(img_path)
        if self.scale != 1:
            width, height = image_pil.size
            new_size = (int(width*self.scale),int(height*self.scale))
            image_pil = image_pil.resize(new_size)
        image = self.trans_to_tensor(image_pil)

        label_json = data_sample["data_path"]
        with open(label_json, "r") as json_file:
            data = json.load(json_file)

        joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                data['sim_state']['joints'][1]['position'],
                                data['sim_state']['joints'][2]['position'],
                                data['sim_state']['joints'][3]['position'],
                                data['sim_state']['joints'][4]['position'],
                                data['sim_state']['joints'][5]['position'],
                                data['sim_state']['joints'][6]['position']])


        joint_angle = torch.tensor(joint_angle, dtype=torch.float)



        return image, joint_angle

    def get_data_with_keypoints(self, idx):

        data_sample = self.ndds_dataset[idx]
        # load image
        img_path = data_sample["image_paths"]["rgb"]
        image_pil = pil_loader(img_path)
        if self.scale != 1:
            width, height = image_pil.size
            new_size = (int(width*self.scale),int(height*self.scale))
            image_pil = image_pil.resize(new_size)
        image = self.trans_to_tensor(image_pil)

        label_json = data_sample["data_path"]
        with open(label_json, "r") as json_file:
            data = json.load(json_file)

        if "panda" in self.data_folder:
            joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                    data['sim_state']['joints'][1]['position'],
                                    data['sim_state']['joints'][2]['position'],
                                    data['sim_state']['joints'][3]['position'],
                                    data['sim_state']['joints'][4]['position'],
                                    data['sim_state']['joints'][5]['position'],
                                    data['sim_state']['joints'][6]['position']])


            joint_angle = torch.tensor(joint_angle, dtype=torch.float)


        else:
            joint_angle = None

        keypoints = data['objects'][0]['keypoints']

        return image, joint_angle, keypoints

