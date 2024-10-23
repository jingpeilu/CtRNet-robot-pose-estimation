#!/usr/bin/env python3
import sys
import os
import warnings

import numpy as np
import time
import rospy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs
import geometry_msgs
import kornia
import shutil
from scipy.io import loadmat
import json

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

import matplotlib.patches as mpatches
#os.environ['ROS_MASTER_URI']='http://192.168.1.116:11311'
#os.environ['ROS_IP']='192.168.1.186'
import calibrate_panda
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
args.scale = 0.3125
args.height = 1536
args.width = 2048
args.fx, args.fy, args.px, args.py = 967.2597045898438, 967.2623291015625, 1024.62451171875, 772.18994140625
# args.fx, args.fy, args.px, args.py = 960.41357421875, 960.22314453125, 1021.7171020507812, 776.2381591796875
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
    return image, image_pil

#############################################################################3

#start = time.time()
visual_idx = 0

new_data = False
points_2d = None
image_pil = None
joint_angles = None
cTr = None
joint_confidence = None
image_msg = None
def gotData(img_msg, joint_msg):
    #global start
    global new_data, points_2d, image_pil, joint_angles, cTr, joint_confidence, image_msg
    image_msg = img_msg
    # print("Received data!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        image, image_pil = preprocess_img(cv_img,args)

        joint_angles = np.array(joint_msg.position)[[0,1,2,3,4,5,6]]

        if args.use_gpu:
            image = image.cuda()

        cTr, points_2d, segmentation, joint_confidence = CtRNet.inference_single_image(image, joint_angles)

        
        #### visualization code ####
        #points_2d = points_2d.detach().cpu().numpy()
        #img_np = to_numpy_img(image)
        #img_np = overwrite_image(img_np,points_2d[0].astype(int), color=(1,0,0))
        #plt.imsave("test.png",img_np)
        ####
    except CvBridgeError as e:
        print(e)
    new_data = True

def save_data(ep_info, ep_name, ep_path, images, joint_angles, timestamps):
    print("saving data...")
    for i, timestamp in enumerate(timestamps):
        curr_image = images[i]
        image_num = str(i).zfill(4)
        image_file_name = f"{ep_name}_f{image_num}.png"
        image_path = os.path.join(ep_path, image_file_name)
        curr_image.save(image_path)

        step_obj = {"img_file": image_file_name, "joint_angles": joint_angles[i].tolist()}
        ep_info["steps"][timestamp] = step_obj

    json_path = os.path.join(ep_path, "info.json")
    with open(json_path, "w") as f:
        json.dump(ep_info, f, indent=4)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="`XYZW` quaternion coefficient order is deprecated and will be removed after > 0.6. Please use `QuaternionCoeffOrder.WXYZ` instead.")

    rospy.init_node('capture_data')

    ep_info = {"calibration_info": {}, "steps": {}}
    image_topic = "/rgb/image_raw"
    robot_joint_topic = "/joint_states"
 
    image_sub = Subscriber(image_topic, sensor_msgs.msg.Image)
    robot_j_sub = Subscriber(robot_joint_topic, sensor_msgs.msg.JointState)

    ats = ApproximateTimeSynchronizer([image_sub, robot_j_sub], queue_size=10, slop=5)
    ats.registerCallback(gotData)
    dataset_dir = os.path.join(args.base_dir, "panda_dataset")
    # Create episode folder
    ep_num = str(len(os.listdir(dataset_dir))+1).zfill(4)
    ep_name = f"panda_{ep_num}"
    ep_path = os.path.join(dataset_dir, ep_name)
    os.mkdir(ep_path)

    # Calibrate panda
    ref_ep = None
    while True:
        recalib = input("Recalibrate camera-to-base? [y/n] ")
        if recalib == "y":
            calibrate_panda.main()
            ref_ep = ep_name
            break
        elif recalib == "n":
            ref_ep = input("Enter name of another episode with the same calibration file (Enter for previous episode): ")
            if ref_ep == "":
                prev_ep = str(ep_num - 1).zfill(4)
                ref_ep = f"panda_c_{prev_ep}"
            break
        else:
            continue
    
    matlab_output_dir = "/home/matlab_calibrate/output"
    ref_output_dir = os.path.join(matlab_output_dir, ref_ep)
    calib_cTr_mat_path = os.path.join(ref_output_dir, "calib_ctr.mat")
    calib_cTr = loadmat(calib_cTr_mat_path)["TBaseAA"]

    ep_info["calibration_info"]["extrinsic"] = calib_cTr.tolist()
    ep_info["calibration_info"]["intrinsic"] = np.float64(CtRNet.intrinsics).tolist()

    all_images = []
    all_joint_angles = []
    all_timestamps = []
    input("Prepare to move panda. During capture Ctrl+C to stop and save data. Enter to start.")
    # Main loop:
    rate = rospy.Rate(30) # 30hz
    while not rospy.is_shutdown():
        try:
            if new_data:
                # Copy all new data gotData
                # new_points_2d = torch.clone(points_2d)
                new_image = image_pil
                new_joint_angles = np.copy(joint_angles)
                new_timestamp = rospy.get_time()

                all_images.append(new_image)
                all_joint_angles.append(new_joint_angles)
                all_timestamps.append(new_timestamp)

            rate.sleep()
        except KeyboardInterrupt:
            rospy.signal_shutdown("Done collecting data.")

    save_data(ep_info, ep_name, ep_path, all_images, all_joint_angles, all_timestamps)

