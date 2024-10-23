# Markerless Camera-to-Robot Pose Estimation via Self-supervised Sim-to-Real Transfer

PyTorch implementation of CtRNet https://arxiv.org/abs/2302.14332


### Dependencies
Recommend set up the environment using Anaconda.
Code is developed and tested on Ubuntu 20.04.
- Python(3.8)
- Numpy(1.22.4)
- PyTorch(1.10.0)
- torchvision(0.11.1)
- pytorch3d(0.6.2)
- Kornia(0.6.3)
- Transforms3d(0.3.1)

More details see `environment.yml`.

## Usage
- See `inference_single_frame.ipynb` for example single frame inference.
- We provide ROS node for CtRNet, which subscribes image and joint state topics and publishes robot pose.
```
python ros_node/panda_pose.py
```

## Dataset

1. [DREAM dataset](https://github.com/NVlabs/DREAM/blob/master/data/DOWNLOAD.sh)
2. [Baxter dataset](https://drive.google.com/file/d/12bCv6GBuh-FdvLGKjlUx2jPN-DBRUqUn/view?usp=share_link)

## Weights

Weights for Panda and Baxter can be found [here](https://drive.google.com/file/d/1OAamxl3_cMLdlpksNo0p-8K20fSMukbI/view?usp=share_link).


## Videos
Using CtRNet for visual servoing experiment with moving camera.



https://user-images.githubusercontent.com/22925036/230662315-5ca62a10-e6b5-4eee-8abc-c3bf2b0fbaae.mp4



https://user-images.githubusercontent.com/22925036/230662322-89b90a32-ca7b-4f64-a6ad-f9b807ddc08d.mp4

## Changes implemented by Tristin
### Refactor to ROS package
* `setup.py` lets ROS know where the directories are
* `base_dir` is now `/home/workspace/src/ctrnet-robot-pose-estimation-ros`
* the original git repository is now made to work in a ROS workspace with a build, devel, and src directories. Remember to `source devel/setup.bash` before use

Outside of docker environment rosbuild was used, inside of docker container it is built with catkin_make
```
rosrun ctrnet-robot-pose-estimation-ros panda_pose.py
```
### Updated DockerFile
Based on a general ROS Docker container with torch3d installed.
```
#!/bin/bash

docker run -it --rm \
        -v ${PWD}/ctrnet_ws:/home/workspace \
        -v ${PWD}/dataset:/home/dataset \
        --net="host" \
        -e ROS_IP=${ROS_IP} \
        -e ROS_MASTER_URI=${ROS_MASTER_URI} \
        --gpus=all \
        rpe:ros_torch3d \
        bash
```
### Filtering
* Did basic refactoring and value passing through functions to get the below to work as intended
* XYZ and quaternion components of the camera to robot (ctr) matrix is separted when calculating the statistical values
* Heatmap confidence filtering: takes top 40 values for each joint prediction softmax to get confidence for each joint. When the confidence of `joint_confident_thresh` or more amount is below `0.90` we skip that ctr prediction
* z-score ((x - sample_mean) / sd)
  * Activate by `filtering_method = "z-score"` (default `filtering_method = "none"`)
  * Takes a 30 samples and compares future samples with the dataset, outliers don't get published, inliers always get published and has a increasing probability (uses an min function on an adjusted square root function to get this result) to be added to the dataset when the max z-score of the ctr gets larger
* modified z-score ((x - sample_mean) / MAD)
  * Activate by `filtering_method = "mod_z_score"`
  * Similar idea to z-score filtering, uses Median Absolute Deviation
  * Is supposed to be less influnced by outliers in the dataset
### Results
* ctrnet can now be run as a rosnode
* confidence filtering removes lots of jumping poses from happening when the camera loses track of joints (i.e. object obstructs the joint)
* z-score and modified z-score removes lots of outliers from being published, lots of hyperparamters to adjust for optimal results
* Most testing has been done with moving the panda arm and not the camera
* Using z-score and modified z-score has the weakness of not being able to deal with robot movement well, probablistic adding and removing data points to the dataset works after a long delay. Will try to resolve this issue with Kalman filtering