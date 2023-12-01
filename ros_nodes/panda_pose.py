#!/usr/bin/env python3
import sys
import os
base_dir = os.path.abspath(".")
sys.path.append(base_dir)

import numpy as np
import time
import rospy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs
import geometry_msgs
import kornia

import torch
import torchvision.transforms as transforms

from PIL import Image as PILImage
from utils import *
from models.CtRNet import CtRNet

import cv2
bridge = CvBridge()

import transforms3d as t3d
import tf2_ros
from sklearn.metrics.pairwise import rbf_kernel
#os.environ['ROS_MASTER_URI']='http://192.168.1.116:11311'
#os.environ['ROS_IP']='192.168.1.186'

################################################################
import argparse
parser = argparse.ArgumentParser()

args = parser.parse_args("")

args.base_dir = "/home/workspace/src/ctrnet-robot-pose-estimation-ros/"
args.use_gpu = True
args.trained_on_multi_gpus = True
args.keypoint_seg_model_path = os.path.join(args.base_dir,"weights/panda/panda-3cam_azure/net.pth")
args.urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")
args.robot_name = 'Panda' # "Panda" or "Baxter_left_arm"
args.n_kp = 7
args.scale = 0.15625
args.height = 1536
args.width = 2048
args.fx, args.fy, args.px, args.py = 960.41357421875, 960.22314453125, 1021.7171020507812, 776.2381591796875
# scale the camera parameters
args.width = int(args.width * args.scale)
args.height = int(args.height * args.scale)
args.fx = args.fx * args.scale
args.fy = args.fy * args.scale
args.px = args.px * args.scale
args.py = args.py * args.scale

if args.use_gpu:
    device = "cuda"
else:
    device = "cpu"

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CtRNet = CtRNet(args)

def preprocess_img(cv_img,args):
    image_pil = PILImage.fromarray(cv_img)
    width, height = image_pil.size
    new_size = (int(width*args.scale),int(height*args.scale))
    image_pil = image_pil.resize(new_size)
    image = trans_to_tensor(image_pil)
    return image


#############################################################################3

#start = time.time()
point_samples = []
rotation_samples = []
prev_confidence = []
T_minus_one = None
def gotData(img_msg, joint_msg):
    #global start
    global point_samples
    global rotation_samples
    global prev_confidence
    global T_minus_one
    # print("Received data!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        image = preprocess_img(cv_img,args)

        joint_angles = np.array(joint_msg.position)[[0,1,2,3,4,5,6]]

        if args.use_gpu:
            image = image.cuda()

        cTr, points_2d, segmentation, confidence = CtRNet.inference_single_image(image, joint_angles)
        # print(cTr)
        # update_publisher(cTr, img_msg)
        qua = kornia.geometry.conversions.angle_axis_to_quaternion(cTr[:,:3]).detach().cpu() # xyzw
        T = cTr[:,3:].detach().cpu()
        # print(T)
        # print(cTr.shape)
        # print(T.shape)
        # print(qua.shape)
        filtering_method = "particle"
        joint_confident_thresh = 3
        num_joint_confident = torch.sum(torch.gt(confidence, 0.90))
        if filtering_method == "none":
            update_publisher(cTr, img_msg, qua.numpy().squeeze(), T.numpy().squeeze())
            return
        elif filtering_method == "particle":
            if T_minus_one is not None:
                pred_T = particle_filter(cTr, T_minus_one, T, joint_angles, 0.01, 1000)
                # print(T)
                # print(pred_T)
                update_publisher(cTr, img_msg, qua.numpy().squeeze(), pred_T.numpy().squeeze())
            else:
                print(f"Skipping because t-1 is {T_minus_one}")
            T_minus_one = T
            return
        # avg_confidence = torch.mean(confidence)
        # print(avg_confidence)
        if num_joint_confident < joint_confident_thresh:
            print(f"Only confident with {num_joint_confident} joints, skipping...")
            return

        if len(point_samples) < 30:
            print(f"Number of samples: {len(point_samples)}")
            point_samples.append(T)
            rotation_samples.append(qua)
            prev_confidence = confidence
            return
        
        # confidence_diff = prev_confidence - confidence
        # print(confidence_diff)

        is_not_outlier = None
        thresh = 1.5
        if filtering_method == "z_score":
            is_not_outlier, max_pos_z, max_rot_z = z_score(T, qua, thresh)
        elif filtering_method == "mod_z_score":
            is_not_outlier, max_pos_z, max_rot_z = mod_z_score(T, qua, thresh)
        else:
            print("Invalid filtering method")

        keep_outlier_prob = torch.min(torch.tensor(1), torch.sqrt(torch.max(max_pos_z, max_rot_z) - 1) / 3)
        keep_outlier = random.random() < keep_outlier_prob
        if is_not_outlier:
            print("Kept cTr")
            point_samples.pop(0)
            rotation_samples.pop(0)
            point_samples.append(T)
            rotation_samples.append(qua)
            update_publisher(cTr, img_msg, qua.numpy().squeeze(), T.numpy().squeeze())
        elif keep_outlier:
            print("Skipped but kept outlier")
            # print(keep_outlier_prob)
            point_samples.pop(0)
            rotation_samples.pop(0)
            point_samples.append(T)
            rotation_samples.append(qua)
        else:
            print("Skipped cTr")

        #### visualization code ####
        #points_2d = points_2d.detach().cpu().numpy()
        #img_np = to_numpy_img(image)
        #img_np = overwrite_image(img_np,points_2d[0].astype(int), color=(1,0,0))
        #plt.imsave("test.png",img_np)
        ####


    except CvBridgeError as e:
        print(e)
def particle_filter(cTr, T_minus_one, T, joint_angles, sigma, m):
    # print("--------------------------------------")
    print("Particle filter")

    # Step 1
    normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([sigma]))
    omega = normal.sample((m, T_minus_one.shape[1])).squeeze()
    # print(T_minus_one)
    # print(omega)
    # print(T.repeat(m,1))

    # m x 3 tensor of particles
    particles = T_minus_one.repeat(m,1) + omega

    # Step 2
    # print(CtRNet.intrinsics)
    # print(particles[1, :].numpy())
    # print(f"joint_angles: {joint_angles}")
    _,t_list = CtRNet.robot.get_joint_RT(joint_angles)
    points_3d = torch.from_numpy(np.array(t_list)).float()
    # print(points_3d)
    # if self.args.robot_name == "Panda":
    #     points_3d = points_3d[[0,2,3,4,6,7,8]] # remove 1 and 5 links as they are overlapping with 2 and 6
    #     # Rotating to ROS format
    p_t = torch.vstack((points_3d.T, torch.ones((1, points_3d.shape[0]))))
    
    ctr_particle_batch = []

    for i in range(m):
        # print(f"adding {i} particle")
        cvTr = np.eye(4)
        cvTr[:3, :3] = kornia.geometry.conversions.angle_axis_to_rotation_matrix(cTr[:, :3]).detach().cpu().numpy().squeeze()
        cvTr[:3, 3] = particles[i,:]
        ctr_particle_batch.append(cvTr)
    ctr_particle_batch = np.array(ctr_particle_batch)
    cvTr_pt = torch.from_numpy(ctr_particle_batch).float() @ p_t
    # print(cvTr_pt.shape)
    K = torch.from_numpy(CtRNet.intrinsics).float()
    K_cvtr_pt = K @ cvTr_pt[:, :3, :]
    z_t_hats = (1/torch.pi) * K_cvtr_pt
    # print(cvTr @ points_3d)
    # print((torch.from_numpy(CtRNet.intrinsics).float() @ (particles[1, :] @ points_3d)))

    # print((1/torch.pi) * (torch.from_numpy(CtRNet.intrinsics).float() @ particles[1, :]))
    # Step 3
    cvTr_gt = np.eye(4)
    cvTr_gt[:3, :3] = kornia.geometry.conversions.angle_axis_to_rotation_matrix(cTr[:, :3]).detach().cpu().numpy().squeeze()
    cvTr_gt[:3, 3] = T
    cvTr_gt_pt = torch.from_numpy(cvTr_gt).float() @ p_t
    K_cvtr_gt_pt = K @ cvTr_gt_pt[:3, :]
    z_t = (1/torch.pi) * K_cvtr_gt_pt
    z_t = z_t[:2, :].reshape(1, -1)

    z_t_hats = z_t_hats[:, :2, :]
    z_t_hats = z_t_hats.reshape(m, -1)
    # print(z_t_hats)
    # print(z_t)
    w = rbf_kernel(z_t_hats, Y=z_t)
    # print(w.shape)
    # print(ctr_particle_batch.shape)
    w_ctr = w[:, :, None] * ctr_particle_batch
    sum_w_ctr = torch.sum(torch.from_numpy(w_ctr), 0)
    sum_w = torch.sum(torch.from_numpy(w))
    pred_ctr = sum_w_ctr / sum_w
    # print("--------------------------------------")
    # print(pred_ctr)
    # print(cvTr_gt)
    pred_pos = pred_ctr[:3, 3:].T
    return pred_pos

    

    # # ROS camera to CV camera transform
    # cTcv = np.array([[0, 0 , 1, 0], [-1, 0, 0 , 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    # print(cTcv)
    # T = cTcv@cvTr
    # print(T)
    # qua = t3d.quaternions.mat2quat(T[:3, :3]) # wxyz
    
    # from robot_arm import PandaArm
    # self.robot = PandaArm(args.urdf_file)
    # self.robot.fkine()

def z_score(T, qua, thresh):
    point_mean = torch.mean(torch.stack(point_samples), dim=0)
    point_std = torch.std(torch.stack(point_samples), dim=0)
    rotation_mean = torch.mean(torch.stack(rotation_samples), dim=0)
    rotation_std = torch.std(torch.stack(rotation_samples), dim=0)
    point_z_score  = (T - point_mean) / point_std
    rotation_z_score  = (qua - rotation_mean) / rotation_std

    test_keep_stable_point = not torch.any(torch.gt(torch.abs(point_z_score), thresh))
    test_keep_stable_rotation = not torch.any(torch.gt(torch.abs(rotation_z_score), thresh))
    if not (test_keep_stable_point and test_keep_stable_rotation):
        print(f"test points z {point_z_score}") 
        print(f"test rotations z {rotation_z_score}") 
    return (test_keep_stable_point and test_keep_stable_rotation), torch.max(torch.abs(point_z_score)), torch.max(torch.abs(rotation_z_score))

def median_abs_deviation(data):
    data_stack = torch.stack(data, dim=0)
    data_mean = torch.mean(data_stack)
    data_mean_repeated = data_mean.unsqueeze(0).repeat(data_stack.shape[0],1)
    mad = torch.median(torch.abs(data_stack - data_mean_repeated))
    return mad

def mod_z_score(T, qua, thresh):
    point_mean = torch.mean(torch.stack(point_samples), dim=0)
    point_mad = median_abs_deviation(point_samples)
    rotation_mean = torch.mean(torch.stack(rotation_samples), dim=0)
    rotation_mad = median_abs_deviation(rotation_samples)

    point_mod_z_score  = (0.6745*(T - point_mean)) / point_mad
    rotation_mod_z_score  = (0.6745*(qua - rotation_mean)) / rotation_mad

    test_keep_stable_point = not torch.any(torch.gt(torch.abs(point_mod_z_score), thresh))
    test_keep_stable_rotation = not torch.any(torch.gt(torch.abs(rotation_mod_z_score), thresh))
    if not (test_keep_stable_point and test_keep_stable_rotation):
        print(f"test points z {point_mod_z_score}") 
        print(f"test rotations z {rotation_mod_z_score}") 
    return (test_keep_stable_point and test_keep_stable_rotation), torch.max(torch.abs(point_mod_z_score)), torch.max(torch.abs(rotation_mod_z_score))

def update_publisher(cTr, img_msg, qua, T):
    p = geometry_msgs.msg.PoseStamped()
    p.header = img_msg.header
    p.pose.position.x = T[0]
    p.pose.position.y = T[1]
    p.pose.position.z = T[2]
    p.pose.orientation.x = qua[0]
    p.pose.orientation.y = qua[1]
    p.pose.orientation.z = qua[2]
    p.pose.orientation.w = qua[3]
    #print(p)
    pose_pub.publish(p)

    # TODO: Not using the filtered output!
    # Rotating to ROS format
    cvTr= np.eye(4)
    cvTr[:3, :3] = kornia.geometry.conversions.angle_axis_to_rotation_matrix(cTr[:, :3]).detach().cpu().numpy().squeeze()
    cvTr[:3, 3] = np.array(cTr[:, 3:].detach().cpu())

    # ROS camera to CV camera transform
    cTcv = np.array([[0, 0 , 1, 0], [-1, 0, 0 , 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    T = cTcv@cvTr
    qua = t3d.quaternions.mat2quat(T[:3, :3]) # wxyz
    # Publish Transform
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "camera_base"
    t.child_frame_id = "panda_link0"
    t.transform.translation.x = T[0, 3]
    t.transform.translation.y = T[1, 3]
    t.transform.translation.z = T[2, 3]
    t.transform.rotation.x = qua[1]
    t.transform.rotation.y = qua[2]
    t.transform.rotation.z = qua[3]
    t.transform.rotation.w = qua[0]
    br.sendTransform(t)

rospy.init_node('panda_pose')
# Define your image topic
image_topic = "/rgb/image_raw"
robot_joint_topic = "/joint_states"
robot_pose_topic = "robot_pose"
# Set up your subscriber and define its callback
#rospy.Subscriber(image_topic, sensor_msgs.msg.Image, gotData)

image_sub = Subscriber(image_topic, sensor_msgs.msg.Image)
robot_j_sub = Subscriber(robot_joint_topic, sensor_msgs.msg.JointState)
pose_pub = rospy.Publisher(robot_pose_topic, geometry_msgs.msg.PoseStamped, queue_size=1)

ats = ApproximateTimeSynchronizer([image_sub, robot_j_sub], queue_size=10, slop=5)
ats.registerCallback(gotData)


# Main loop:
rate = rospy.Rate(30) # 30hz

while not rospy.is_shutdown():
    rate.sleep()