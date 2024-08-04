import sys
import os
base_dir = os.path.abspath(".")
sys.path.append(base_dir)

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from pathlib import Path

from utils import *

import argparse
from imageloaders.DREAM import ImageDataLoaderReal, load_camera_parameters
from models.CtRNet import CtRNet
from models.heatmap import heatmap_to_keypoints
from models.BPnP import batch_project
from evaluation.evaluate import evaluate_ADD

def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.data_folder = '/media/jingpei/DATA/DREAM/data/real/panda-orb'
    args.base_dir = "/home/jingpei/Desktop/CtRNet-robot-pose-estimation"
    args.use_gpu = True
    args.trained_on_multi_gpus = True
    args.keypoint_seg_model_path = os.path.join(args.base_dir,"outputs/pretrain/net_best.pth")
    #args.keypoint_seg_model_path = os.path.join(args.base_dir,"weights/baxter/net.pth")
    args.urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")

    ##### training parameters #####
    args.batch_size = 32
    args.num_workers = 8
    args.lr = 1e-6
    args.beta1 = 0.9
    args.n_epoch = 500
    args.out_dir = 'outputs/panda-orb'
    args.ckp_per_epoch = 10
    args.reproj_err_scale = 1.0 / 100.0
    ################################

    args.robot_name = 'Panda' # "Panda" or "Baxter_left_arm"
    args.n_kp = 12
    args.scale = 0.5
    args.height = 480
    args.width = 640
    args.fx, args.fy, args.px, args.py = load_camera_parameters(args.data_folder)
    

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    return args

args = get_args()
if args.out_dir:
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

print(args)


trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


datasets = {}
dataloaders = {}
data_n_batches = {}
for phase in ['train','valid']:
    if phase == 'train':
        datasets[phase] = ImageDataLoaderReal(data_folder = ["/media/jingpei/DATA/DREAM/data/real/panda-3cam_kinect360", 
                                                            "/media/jingpei/DATA/DREAM/data/real/panda-3cam_azure", 
                                                            "/media/jingpei/DATA/DREAM/data/real/panda-3cam_realsense", 
                                                            "/media/jingpei/DATA/DREAM/data/real/panda-orb"], 
                                                            scale = args.scale, trans_to_tensor = trans_to_tensor)
    else:
        datasets[phase] = ImageDataLoaderReal(data_folder = args.data_folder, scale = args.scale, trans_to_tensor = trans_to_tensor)

    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers)

    data_n_batches[phase] = len(dataloaders[phase])


######## setup CtRNet ########
from models.CtRNet import CtRNet

CtRNet = CtRNet(args)

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


criterionMSE_sum = torch.nn.MSELoss(reduction='sum')
criterionMSE_mean = torch.nn.MSELoss(reduction='mean')
criterionBCE = torch.nn.BCEWithLogitsLoss()

optimizer = optim.Adam(CtRNet.keypoint_seg_predictor.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)


if args.use_gpu:
    criterionMSE_sum = criterionMSE_sum.cuda()
    criterionMSE_mean = criterionMSE_mean.cuda()
    criterionBCE = criterionBCE.cuda()
    device = "cuda"
else:
    device = "cpu"

epoch_writer = SummaryWriter(comment="_writter")

best_valid_loss = np.inf
best_AUC_ADD = 0

for epoch in range(0, args.n_epoch):
    phases = ['train','valid']

    for phase in phases:

        # train keypoint detector
        
        CtRNet.keypoint_seg_predictor.train(phase == 'train')

        loader = dataloaders[phase]


        if phase == 'train':
            iter_writer = SummaryWriter(comment="_epoch_" + str(epoch) + "_" + phase)

            meter_loss = AverageMeter()
            meter_loss_mse = AverageMeter()
            meter_loss_bce = AverageMeter()

            for i, data in tqdm(enumerate(loader), total=data_n_batches[phase]):

                if args.use_gpu:
                    if isinstance(data, list):
                        data = [d.cuda() for d in data]
                    else:
                        data = data.cuda()

                # load data
                img, joint_angle = data

                with torch.set_grad_enabled(phase == 'train'):

                    # detect 2d keypoints
                    heatmap, segmentation = CtRNet.keypoint_seg_predictor(img)
                    points_2d = heatmap_to_keypoints(heatmap, temperature=1e-2)

                    mask_list = list()
                    seg_weight_list = list()
                    points_2d_proj_list = list()
                    good_sample_idx = list()
                    for b in range(img.shape[0]):
                        # compute 3d keypoints
                        points = CtRNet.robot.get_3d_keypoints(joint_angle[b].cpu().squeeze())
                        points_3d = torch.from_numpy(points).float().to(CtRNet.device)

                        # solve for pose
                        #init_pose = torch.tensor([[ 1.6915,  0.4521, -0.3850, -0.2003,  0.3975,  1.0556]])
                        #cTb = bpnp(points_2d[b][None], points_3d, K, init_pose)
                        cTr = CtRNet.bpnp(points_2d[b][None], points_3d, CtRNet.K)

                        # config robot mesh
                        robot_mesh = robot_renderer.get_robot_mesh(joint_angle[b].cpu().squeeze())


                        # render robot mask
                        rendered_image = CtRNet.render_single_robot_mask(cTr.squeeze(), robot_mesh, robot_renderer)

                        mask_list.append(rendered_image)

                        points_2d_proj = batch_project(cTr, points_3d, CtRNet.K)
                        
                        reproject_error = criterionMSE_mean(points_2d[b], points_2d_proj.squeeze())
                        scale = 1.0 / 10.0
                        seg_weight = torch.exp(-reproject_error * scale)
                        seg_weight_list.append(seg_weight)

                        points_2d_proj_list.append(points_2d_proj)
                        
                        ############### debug #############################
                        if torch.abs(torch.sum(rendered_image)) < 10 or torch.abs(torch.sum(rendered_image)) > 70000 or reproject_error > 100:
                            #print("bad rendering")
                            #print(T)
                            iter_writer.add_image('[bad rendering] rendered vs segmentation', np.concatenate((rendered_image.squeeze().cpu().detach().numpy(),
                                                                            torch.sigmoid(segmentation[b]).squeeze().cpu().detach().numpy()),
                                                                            axis=1), i, dataformats='HW')
                            img_np = to_numpy_img(img[b])
                            img_np = overwrite_image(img_np,points_2d[b].detach().cpu().numpy().squeeze().astype(int), color=(0,255,0))
                            iter_writer.add_image('[bad rendering] keypoints', img_np, i, dataformats='HWC')

                            #print(seg_weight)
                            #print(reproject_error)
                            #print("-----------------------")
                        else:
                            #points_2d_proj = BPnP.batch_project(cTb, points_3d, K)
                            #reproject_error = criterionMSE_mean(points_2d[b], points_2d_proj)
                            #print(seg_weight)
                            
                            good_sample_idx.append(b)

                        #########################################################
                        

                    mask_batch = torch.cat(mask_list,0)
                    points_2d_proj_batch = torch.cat(points_2d_proj_list,0)

                    img_ref = torch.sigmoid(segmentation).detach()


                    #### using seg_weight_list
                    loss_bce = 0
                    for b in range(segmentation.shape[0]):
                        loss_bce = loss_bce + seg_weight_list[b] * criterionBCE(segmentation[b].squeeze(), mask_batch[b].detach())

                    loss_mse = criterionMSE_sum(mask_batch, img_ref.squeeze())
                    #loss_mse = 0
                    #for b in range(mask_batch.shape[0]):
                    #    loss_mse = loss_mse + seg_weight_list[b] * criterionMSE_sum(mask_batch[b], img_ref[b].squeeze())


                    ### using good_sample_idx
                    #loss_mse = criterionMSE_sum(mask_batch[good_sample_idx], img_ref[good_sample_idx].squeeze())
                    #loss_bce = criterionBCE(segmentation[good_sample_idx].squeeze(), mask_batch[good_sample_idx].detach())


                    #loss_reproj = criterionMSE_mean(points_2d, points_2d_proj_batch)

                    loss = 0.001 * loss_mse + loss_bce

                    meter_loss.update(loss.item(), n=img.size(0))
                    meter_loss_mse.update(loss_mse.item(), n=img.size(0))
                    meter_loss_bce.update(loss_bce.item(), n=img.size(0))

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(CtRNet.keypoint_seg_predictor.parameters(), 10)
                    optimizer.step()

                # write to log

                iter_writer.add_scalar('loss_mse', loss_mse.item(), i)
                iter_writer.add_scalar('loss_bce', loss_bce.item(), i)
                iter_writer.add_scalar('loss_all', loss.item(), i)

                if i % 100 == 0:

                    img_np = to_numpy_img(img[0])
                    img_np_pred = overwrite_image(img_np.copy(),points_2d[0].detach().cpu().numpy().squeeze().astype(int), color=(0,255,0))
                    iter_writer.add_image('[keypoint] predict', img_np_pred, i, dataformats='HWC')
                    iter_writer.add_image('[segmentation] rendered vs segmentation', np.concatenate((mask_batch[0].squeeze().cpu().detach().numpy(),
                                                                            torch.sigmoid(segmentation[0]).squeeze().cpu().detach().numpy()),
                                                                            axis=1), i, dataformats='HW')

            log = '%s [%d/%d] Loss: %.6f, LR: %f' % (
                phase, epoch, args.n_epoch,
                meter_loss.avg,
                get_lr(optimizer))

            iter_writer.close()

            print(log)

            epoch_writer.add_scalar('loss_bce', meter_loss_bce.avg, epoch)
            epoch_writer.add_scalar('loss_mse', meter_loss_mse.avg, epoch)
            epoch_writer.add_scalar('loss_all', meter_loss.avg, epoch)

            scheduler.step(meter_loss.avg)


        if phase == 'valid':


            CtRNet.keypoint_seg_predictor.train(phase == 'train')

            AUC_ADD, mean_error = evaluate_ADD(CtRNet, datasets[phase])


            epoch_writer.add_scalar('AUC_ADD', AUC_ADD, epoch)
            epoch_writer.add_scalar('mean error', mean_error, epoch)

            if AUC_ADD > best_AUC_ADD:
                best_AUC_ADD = AUC_ADD

                torch.save(CtRNet.keypoint_seg_predictor.state_dict(), '%s/net_best.pth' % (args.out_dir))

            log = 'Best AUC_ADD: %.6f' % (best_AUC_ADD)
            print(log)
            
            if epoch % args.ckp_per_epoch == 0:
                torch.save(CtRNet.keypoint_seg_predictor.state_dict(), '%s/net_epoch_%d.pth' % (args.out_dir, epoch))

            
epoch_writer.close()



