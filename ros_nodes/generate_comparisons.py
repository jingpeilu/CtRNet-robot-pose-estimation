import sys
import os
base_dir = "/home/workspace/src/ctrnet-robot-pose-estimation-ros"
sys.path.append(base_dir)

import numpy as np
from scipy.io import loadmat
from utils import *

import torch

from utils import *
from models.CtRNet import CtRNet

import pickle
import matplotlib.patches as mpatches

################################################################
import argparse
parser = argparse.ArgumentParser()

args = parser.parse_args("")

args.base_dir = "/home/workspace/src/ctrnet-robot-pose-estimation-ros/"
args.use_gpu = False
args.trained_on_multi_gpus = True
args.keypoint_seg_model_path = os.path.join(args.base_dir,"weights/panda/panda-3cam_azure/net.pth")
args.urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")
args.robot_name = 'Panda' # "Panda" or "Baxter_left_arm"
args.n_kp = 7
args.scale = 0.15625
args.height = 1536
args.width = 2048
# args.fx, args.fy, args.px, args.py = 960.41357421875, 960.22314453125, 1021.7171020507812, 776.2381591796875
args.fx, args.fy, args.px, args.py = 967.2597045898438, 967.2623291015625, 1024.62451171875, 772.18994140625

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

CtRNet = CtRNet(args)

matlab_output_dir = "/home/matlab_calibrate/output"
matlab_calib_dir = "/home/matlab_calibrate"
#############################################################################3

def read_data_dir_and_write_figure(data_dir_name):
    curr_calib_dir = os.path.join(matlab_calib_dir, data_dir_name)
    curr_output_dir = os.path.join(matlab_output_dir, data_dir_name)
    image_dir = os.path.join(curr_calib_dir, "images")
    pickle_path = os.path.join(curr_calib_dir, "all_data.pkl")
    all_data = pickle.load(open(pickle_path, "rb"))

    mesh_files = [base_dir + "/urdfs/Panda/meshes/visual/link0/link0.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link1/link1.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link2/link2.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link3/link3.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link4/link4.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link5/link5.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link6/link6.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link7/link7.obj",
              base_dir + "/urdfs/Panda/meshes/visual/hand/hand.obj",
             ]
    robot_renderer = CtRNet.setup_robot_renderer(mesh_files)

    print("Since matlab code might not detect checkerboard on some images, you must match corresponding input and output images.")
    input_img_idx = int(input("Enter input image file name index: "))
    output_img_idx = int(input("Enter output image file name index: "))
    robot_mesh = robot_renderer.get_robot_mesh(all_data["all_joint_angles"][input_img_idx])
    
    calib_cTr_mat_path = os.path.join(curr_output_dir, "calib_ctr.mat")
    calib_cTr = torch.tensor(loadmat(calib_cTr_mat_path)["TBaseAA"])
    calib_rendered_image = CtRNet.render_single_robot_mask(calib_cTr.squeeze(), robot_mesh, robot_renderer)
    ctrnet_rendered_image = CtRNet.render_single_robot_mask(torch.from_numpy(all_data["all_cTr"][input_img_idx].squeeze()), robot_mesh, robot_renderer)
    compare_img_path = os.path.join(curr_output_dir, f"compare_in{input_img_idx}_out{output_img_idx}.png")

    input_img_path = os.path.join(image_dir, f"image_{input_img_idx}.png")
    input_img = plt.imread(input_img_path)

    red = (1,0,0)
    input_img_anno = 0.0* np.ones(input_img.shape) + input_img * 0.6
    input_img_anno = overwrite_image(input_img_anno, all_data["all_points_2d"][input_img_idx].squeeze().astype(int), color=red)


    output_img_path = os.path.join(curr_output_dir, f"outputImage{output_img_idx}.png")
    output_img = plt.imread(output_img_path)

    fig = plt.figure(figsize=(20,20))
    rows, cols = 2, 2

    fig.add_subplot(rows, cols, 1)
    plt.imshow(input_img_anno)
    plt.axis("off")
    plt.title("CtRNet Annotated")
    colors = [red]
    labels = ["points_2d"]
    joint_confidences = all_data["all_joint_confidence"][input_img_idx]
    for i in range(joint_confidences.shape[0]):
        colors.append(tuple(joint_confidences[i]*r for r in red))
        labels.append(f"p{i} conf: {np.round(joint_confidences[i]*100, decimals=1)}")

    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.grid(True)

    fig.add_subplot(rows, cols, 2)
    plt.imshow(output_img)
    plt.axis("off")
    plt.title("Calibration output")

    fig.add_subplot(rows, cols, 3)

    plt.imshow(input_img)
    plt.imshow(ctrnet_rendered_image.squeeze().detach().cpu().numpy(), cmap="RdYlGn", alpha=0.2)
    plt.axis("off")
    plt.title("CtRNet Robot Mask")

    fig.add_subplot(rows, cols, 4)
    plt.imshow(input_img)
    plt.imshow(calib_rendered_image.squeeze().detach().cpu().numpy(), cmap="RdYlGn", alpha=0.2)
    plt.axis("off")
    plt.title("Calibrated Robot Mask")

    fig.savefig(compare_img_path)
    plt.close()

if __name__ == "__main__":
    data_dir_name = input("Input matlab directory name: ")
    read_data_dir_and_write_figure(data_dir_name)
