"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training and evaluation codes for 
3D human body mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import json
import time
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import numpy as np
import cv2
from tore.modeling_fm.bert import BertConfig
from tore.modeling_fm.bert import FastMETRO_Body_Network as METRO_Network
# Frank
from tore.modeling_fm._smpl import SMPL, Mesh
from tore.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from tore.modeling.hrnet.config import config as hrnet_config
from tore.modeling.hrnet.config import update_config as hrnet_update_config
import tore.modeling.data.config as cfg
from tore.datasets.build import make_data_loader

from tore.utils.logger import setup_logger
from tore.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from tore.utils.miscellaneous import mkdir, set_seed
from tore.utils.metric_logger import AverageMeter, EvalMetricsLogger
from tore.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test
from tore.utils.metric_pampjpe import reconstruction_error
from tore.utils.geometric_layers import orthographic_projection
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

def save_scores(args, split, mpjpe, pampjpe, mpve):
    eval_log = []
    res = {}
    res['mPJPE'] = mpjpe
    res['PAmPJPE'] = pampjpe
    res['mPVE'] = mpve
    eval_log.append(res)
    with open(op.join(args.output_dir, split+'_eval_logs.json'), 'w') as f:
        json.dump(eval_log, f)
    logger.info("Save eval scores to {}".format(args.output_dir))
    return

def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs/2.0)  ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mean_per_joint_position_error(pred, gt, has_3d_joints):
    """ 
    Compute mPJPE
    """
    gt = gt[has_3d_joints == 1]
    gt = gt[:, :, :-1]
    pred = pred[has_3d_joints == 1]

    with torch.no_grad():
        gt_pelvis = (gt[:, 2,:] + gt[:, 3,:]) / 2
        gt = gt - gt_pelvis[:, None, :]
        pred_pelvis = (pred[:, 2,:] + pred[:, 3,:]) / 2
        pred = pred - pred_pelvis[:, None, :]
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_vertex_error(pred, gt, has_smpl):
    """
    Compute mPVE
    """
    pred = pred[has_smpl == 1]
    gt = gt[has_smpl == 1]
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl, device):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 
    
def rectify_pose(pose):
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose

def normal_vector_loss(coord_out, coord_gt, face, has_smpl, device):
    coord_out = coord_out[has_smpl == 1]
    coord_gt = coord_gt[has_smpl == 1]
    if len(coord_gt) > 0:
        face = torch.LongTensor(face).cuda()

        v1_out = coord_out[:,face[:,1],:] - coord_out[:,face[:,0],:]
        v1_out = F.normalize(v1_out, p=2, dim=2) # L2 normalize to make unit vector
        v2_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,0],:]
        v2_out = F.normalize(v2_out, p=2, dim=2) # L2 normalize to make unit vector
        v3_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,1],:]
        v3_out = F.normalize(v3_out, p=2, dim=2) # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:,face[:,1],:] - coord_gt[:,face[:,0],:]
        v1_gt = F.normalize(v1_gt, p=2, dim=2) # L2 normalize to make unit vector
        v2_gt = coord_gt[:,face[:,2],:] - coord_gt[:,face[:,0],:]
        v2_gt = F.normalize(v2_gt, p=2, dim=2) # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2) # L2 normalize to make unit vector

        # valid_mask = valid[:,face[:,0],:] * valid[:,face[:,1],:] * valid[:,face[:,2],:]

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))

        loss = torch.cat((cos1, cos2, cos3),1)
        loss = loss.mean()
        return loss
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def edge_length_loss(coord_out, coord_gt, face, has_smpl, device):
    coord_out = coord_out[has_smpl == 1]
    coord_gt = coord_gt[has_smpl == 1]
    if len(coord_gt) > 0:
        face = torch.LongTensor(face).cuda()

        d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True) + 1e-6)
        d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)
        d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)

        d1_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,1],:])**2,2,keepdim=True) + 1e-6)
        d2_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)
        d3_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,1],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)

        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)

        loss = torch.cat((diff1, diff2, diff3), 1)
        loss = loss.mean()
        return loss
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def min_max_normalize(x):
    # input dim: batch_size, token_num, 49
    batch_size, token_num, _ = x.shape
    ep = 1e-12
    res = torch.zeros_like(x)

    for i in range(batch_size):
        for j in range(token_num):
            temp = x[i, j, :]
            temp_min = temp.min()
            temp_max = temp.max()
            temp = (temp - temp_min) / (temp_max - temp_min + ep)
            res[i, j, :] = temp
    return res

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def itp_loss(token_heatmap, mesh_2d, has_smpl):
    heatmap_size = 7
    batch_size, token_num, _ = token_heatmap.shape
    mask_pred = min_max_normalize(token_heatmap)
    mask_pred = mask_pred.reshape(batch_size, token_num, heatmap_size, heatmap_size)
    mesh_2d = ((mesh_2d + 1) * 0.5) * heatmap_size

    x_axis = mesh_2d[:, :, 0]
    x_res1 = torch.where(x_axis>=0, torch.ones_like(x_axis), torch.zeros_like(x_axis))
    x_res2 = torch.where(x_axis<heatmap_size, torch.ones_like(x_axis), torch.zeros_like(x_axis))
    x_res = torch.logical_and(x_res1, x_res2)

    y_axis = mesh_2d[:, :, 1]
    y_res1 = torch.where(y_axis>=0, torch.ones_like(y_axis), torch.zeros_like(y_axis))
    y_res2 = torch.where(y_axis<heatmap_size, torch.ones_like(y_axis), torch.zeros_like(y_axis))
    y_res = torch.logical_and(y_res1, y_res2)

    res = torch.logical_and(x_res, y_res)
    idx = res.nonzero()

    mask_gt = torch.zeros_like(mask_pred[:,0,:,:])
    batch_idx = idx[:,0]
    x_ = mesh_2d[idx[:,0], idx[:,1], :][:,0]
    y_ = mesh_2d[idx[:,0], idx[:,1], :][:,1]

    mask_gt[batch_idx, x_.long(), y_.long()] = 1
    mask_gt = mask_gt.transpose(1,2) 
    mask_gt = mask_gt.unsqueeze(1).repeat(1,token_num,1,1)
    # Peter: don't know why but visualization suggests that this transpose is needed

    mask_pred = mask_pred[has_smpl==1, :, :, :]
    mask_gt = mask_gt[has_smpl==1, :, :, :]
    loss = torch.sum(mask_pred * (1 - mask_gt))
    # Peter: penalizing the inclusion of non-human patch.
    return loss, mask_gt, mask_pred

    

def run(args, train_dataloader, val_dataloader, METRO_model, smpl, mesh_sampler, renderer):
    smpl.eval()
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs
    if iters_per_epoch<1000:
        args.logging_steps = 500
    


    optimizer = torch.optim.AdamW(params=list(METRO_model.parameters()),
                                           lr=args.lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=1e-4)

    # define loss function (criterion) and optimizer
    criterion_2d_keypoints = torch.nn.L1Loss(reduction='none').cuda(args.device)
    criterion_keypoints = torch.nn.L1Loss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)

    if args.distributed:
        METRO_model = torch.nn.parallel.DistributedDataParallel(
            METRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

        logger.info(
                ' '.join(
                ['Local rank: {o}', 'Max iteration: {a}', 'iters_per_epoch: {b}','num_train_epochs: {c}',]
                ).format(o=args.local_rank, a=max_iter, b=iters_per_epoch, c=args.num_train_epochs)
            )

    start_training_time = time.time()
    end = time.time()
    METRO_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_loss_2djoints = AverageMeter()
    log_loss_3djoints = AverageMeter()
    log_loss_vertices = AverageMeter()
    log_eval_metrics = EvalMetricsLogger()
    log_loss_normal_vector = AverageMeter()
    log_loss_edge_length = AverageMeter()
    log_loss_itp = AverageMeter()
    human_model = SMPL()
    mesh_face = human_model.faces

    for iteration, (img_keys, images, annotations) in tqdm(enumerate(train_dataloader)):

        METRO_model.train()
        iteration += 1
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)
        adjust_learning_rate(optimizer, epoch, args)
        data_time.update(time.time() - end)

        images = images.cuda(args.device)
        gt_2d_joints = annotations['joints_2d'].cuda(args.device)
        gt_2d_joints = gt_2d_joints[:,cfg.J24_TO_J14,:]
        has_2d_joints = annotations['has_2d_joints'].cuda(args.device)

        gt_3d_joints = annotations['joints_3d'].cuda(args.device)
        gt_3d_pelvis = gt_3d_joints[:,cfg.J24_NAME.index('Pelvis'),:3]
        gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:] 
        gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :]
        has_3d_joints = annotations['has_3d_joints'].cuda(args.device)

        gt_pose = annotations['pose'].cuda(args.device)
        gt_betas = annotations['betas'].cuda(args.device)
        has_smpl = annotations['has_smpl'].cuda(args.device)
        # mjm_mask = annotations['mjm_mask'].cuda(args.device)
        # mvm_mask = annotations['mvm_mask'].cuda(args.device)

        # generate simplified mesh
        gt_vertices = smpl(gt_pose, gt_betas)
        gt_vertices_sub2 = mesh_sampler.downsample(gt_vertices, n1=0, n2=2)
        gt_vertices_sub = mesh_sampler.downsample(gt_vertices)

        # normalize gt based on smpl's pelvis 
        gt_smpl_3d_joints = smpl.get_h36m_joints(gt_vertices)
        gt_smpl_3d_pelvis = gt_smpl_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
        gt_vertices_sub2 = gt_vertices_sub2 - gt_smpl_3d_pelvis[:, None, :]
            
        # prepare masks for mask vertex/joint modeling
        # mjm_mask_ = mjm_mask.expand(-1,-1,2051)
        # mvm_mask_ = mvm_mask.expand(-1,-1,2051)
        # meta_masks = torch.cat([mjm_mask_, mvm_mask_], dim=1)

        # forward-pass
        pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, token_heatmap = METRO_model(images)

        # normalize gt based on smpl's pelvis 
        gt_vertices_sub = gt_vertices_sub - gt_smpl_3d_pelvis[:, None, :] 
        gt_vertices = gt_vertices - gt_smpl_3d_pelvis[:, None, :]

        # obtain 3d joints, which are regressed from the full mesh
        pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
        pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]

        # obtain 2d joints, which are projected from 3d joints of smpl mesh
        pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
        pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

        torch.set_printoptions(sci_mode=False)

        mesh_2d = orthographic_projection(gt_vertices, pred_camera)

        # compute 3d joint loss  (where the joints are directly output from transformer)

       
        loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints, has_3d_joints, args.device)
        # compute 3d vertex loss
        loss_vertices = ( args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2, has_smpl, args.device) + \
                            args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub, has_smpl, args.device) + \
                            args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl, args.device) )
        # compute 3d joint loss (where the joints are regressed from full mesh)
        loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints, has_3d_joints, args.device)
        # compute 2d joint loss
        loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints, has_2d_joints)  + \
                         keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints, has_2d_joints)
        
        loss_3d_joints = loss_3d_joints + loss_reg_3d_joints

        # itp loss
        loss_itp, mask_gt, mask_pred = itp_loss(token_heatmap, mesh_2d, has_smpl)

        mask_gt = torch.mean(mask_gt, dim=1)
        mask_pred = torch.mean(mask_pred, dim=1)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        soft_epoch = iteration / iters_per_epoch
        itp_loss_weight = args.itp_loss_weight * sigmoid(soft_epoch - 3)

        loss = args.joints_loss_weight*loss_3d_joints + \
                args.vertices_loss_weight*loss_vertices  + args.vertices_loss_weight*loss_2d_joints + itp_loss_weight * loss_itp

        if "true" in args.edge_and_normal_vector_loss:
            loss_normal_vector = normal_vector_loss(pred_vertices, gt_vertices, mesh_face, has_smpl, args.device)
            loss_edge_length = edge_length_loss(pred_vertices, gt_vertices, mesh_face, has_smpl, args.device)

        # we empirically use hyperparameters to balance difference losses
        if "true" in args.edge_and_normal_vector_loss:
            loss += 100 * loss_normal_vector + 1000 * loss_edge_length


        # update logs
        log_loss_2djoints.update(loss_2d_joints.item(), batch_size)
        log_loss_3djoints.update(loss_3d_joints.item(), batch_size)
        log_loss_vertices.update(loss_vertices.item(), batch_size)
        log_loss_itp.update(loss_itp.item(), batch_size)
        log_losses.update(loss.item(), batch_size)
        if "true" in args.edge_and_normal_vector_loss:
            log_loss_edge_length.update(loss_edge_length.item(), batch_size)
            log_loss_normal_vector.update(loss_normal_vector.item(), batch_size)

        # back prop
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_value_(METRO_model.parameters(), clip_value=0.3)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        args.logging_steps = 500
        if iteration % args.logging_steps == 0 or iteration == max_iter:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                ' '.join(
                ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}',]
                ).format(eta=eta_string, ep=epoch, iter=iteration, 
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) 
                + '  loss: {:.4f}, 2d joint loss: {:.4f}, 3d joint loss: {:.4f}, vertex loss: {:.4f}, edge length loss: {:.4f}, normal vector loss: {:.4f}, itp loss: {:.4f} compute: {:.4f}, data: {:.4f}, lr: {:.6f}'.format(
                    log_losses.avg, log_loss_2djoints.avg, log_loss_3djoints.avg, log_loss_vertices.avg, log_loss_edge_length.avg, log_loss_normal_vector.avg, log_loss_itp.avg, batch_time.avg, data_time.avg, 
                    optimizer.param_groups[0]['lr'])
            )

            mask_gt = mask_gt.detach()
            mask_gt = mask_gt.cpu()
            mask_gt = min_max_normalize(mask_gt)
            mask_gt = mask_gt.unsqueeze(0)
            mask_gt = F.interpolate(mask_gt, size=(224, 224), mode='nearest')
            mask_gt = mask_gt.squeeze()

            mask_pred = mask_pred.detach()
            mask_pred = mask_pred.cpu()
            mask_pred = min_max_normalize(mask_pred)
            mask_pred = mask_pred.unsqueeze(0)
            mask_pred = F.interpolate(mask_pred, size=(224, 224), mode='nearest')
            mask_pred = mask_pred.squeeze()

            itp_imgs = []
            for i in range(min(batch_size, mask_gt.shape[0], mask_pred.shape[0], 10)):
                mask_gt_ = mask_gt[i, :, :]
                mask_pred_ = mask_pred[i, :, :]
                visual_imgs = annotations['ori_img'][i].detach()
                visual_imgs = visual_imgs.transpose(0,1)
                visual_imgs = visual_imgs.transpose(1,2)
                visual_imgs = np.asarray(visual_imgs)
                visual_imgs = np.asarray(visual_imgs[:,:,::-1]) # to BGR
                itp_img_gt = show_cam_on_image(visual_imgs, mask_gt_)
                itp_img_pred = show_cam_on_image(visual_imgs, mask_pred_)
                itp_img_gt = np.asarray(itp_img_gt)
                itp_img_pred = np.asarray(itp_img_pred)
                itp_img = np.hstack([itp_img_gt, itp_img_pred])
                itp_imgs.append(itp_img)
            itp_imgs = np.vstack(itp_imgs)


            vis_imgs = visualize_mesh(   renderer,
                                            annotations['ori_img'].detach(),
                                            annotations['joints_2d'].detach(),
                                            pred_vertices.detach(), 
                                            pred_camera.detach(),
                                            pred_2d_joints_from_smpl.detach(),
                                            mesh_2d.detach())

            vis_imgs = np.asarray(vis_imgs)

            if is_main_process()==True:
                stamp = str(epoch) + '_' + str(iteration)
                temp_fname = args.output_dir + 'visual_' + stamp + '.jpg'
                img = np.concatenate((vis_imgs, itp_imgs), axis=1)
                cv2.imwrite(temp_fname, img)


        if iteration % iters_per_epoch == 0:
            val_mPVE, val_mPJPE, val_PAmPJPE, val_count = run_validate(args, val_dataloader, 
                                                METRO_model, 
                                                criterion_keypoints, 
                                                criterion_vertices, 
                                                epoch, 
                                                smpl,
                                                mesh_sampler)

            logger.info(
                ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
                + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}, Data Count: {:6.2f}'.format(1000*val_mPVE, 1000*val_mPJPE, 1000*val_PAmPJPE, val_count)
            )

            if val_PAmPJPE<log_eval_metrics.PAmPJPE:
                log_eval_metrics.update(val_mPVE, val_mPJPE, val_PAmPJPE, epoch)
                checkpoint_dir = save_checkpoint(METRO_model, args, epoch, iteration)
                
        
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    checkpoint_dir = save_checkpoint(METRO_model, args, epoch, iteration)

    logger.info(
        ' Best Results:'
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}, at epoch {:6.2f}'.format(1000*log_eval_metrics.mPVE, 1000*log_eval_metrics.mPJPE, 1000*log_eval_metrics.PAmPJPE, log_eval_metrics.epoch)
    )


def run_eval_general(args, val_dataloader, METRO_model, smpl, mesh_sampler):
    smpl.eval()
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)

    epoch = 0
    if args.distributed:
        METRO_model = torch.nn.parallel.DistributedDataParallel(
            METRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    METRO_model.eval()

    val_mPVE, val_mPJPE, val_PAmPJPE, val_count = run_validate(args, val_dataloader, 
                                    METRO_model, 
                                    criterion_keypoints, 
                                    criterion_vertices, 
                                    epoch, 
                                    smpl,
                                    mesh_sampler)

    logger.info(
        ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f} '.format(1000*val_mPVE, 1000*val_mPJPE, 1000*val_PAmPJPE)
    )
    # checkpoint_dir = save_checkpoint(METRO_model, args, 0, 0)
    return

def run_validate(args, val_loader, METRO_model, criterion, criterion_vertices, epoch, smpl, mesh_sampler):
    batch_time = AverageMeter()
    mPVE = AverageMeter()
    mPJPE = AverageMeter()
    PAmPJPE = AverageMeter()
    # switch to evaluate mode
    METRO_model.eval()
    smpl.eval()
    with torch.no_grad():
        # end = time.time()
        print("validating...")
        for i, (img_keys, images, annotations) in tqdm(enumerate(val_loader)):
            batch_size = images.size(0)
            # compute output
            images = images.cuda(args.device)
            gt_3d_joints = annotations['joints_3d'].cuda(args.device)
            gt_3d_pelvis = gt_3d_joints[:,cfg.J24_NAME.index('Pelvis'),:3]
            gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:] 
            gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :]
            has_3d_joints = annotations['has_3d_joints'].cuda(args.device)

            gt_pose = annotations['pose'].cuda(args.device)
            gt_betas = annotations['betas'].cuda(args.device)
            has_smpl = annotations['has_smpl'].cuda(args.device)

            # generate simplified mesh
            gt_vertices = smpl(gt_pose, gt_betas)
            gt_vertices_sub = mesh_sampler.downsample(gt_vertices)
            gt_vertices_sub2 = mesh_sampler.downsample(gt_vertices_sub, n1=1, n2=2)

            # normalize gt based on smpl pelvis 
            gt_smpl_3d_joints = smpl.get_h36m_joints(gt_vertices)
            gt_smpl_3d_pelvis = gt_smpl_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            gt_vertices_sub2 = gt_vertices_sub2 - gt_smpl_3d_pelvis[:, None, :] 
            gt_vertices = gt_vertices - gt_smpl_3d_pelvis[:, None, :] 

            # forward-pass
            pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, _  = METRO_model(images)

            # obtain 3d joints from full mesh
            pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

            pred_3d_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
            pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

            # measure errors
            if args.oracle == True or args.oracle == "true":
                pred_3d_oracle_pelvis = pred_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
                pred_3d_joints = pred_3d_joints - pred_3d_oracle_pelvis[:, None, :]

                error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices, has_smpl)
                error_joints = mean_per_joint_position_error(pred_3d_joints, gt_3d_joints,  has_3d_joints)
                error_joints_pa = reconstruction_error(pred_3d_joints.cpu().numpy(), gt_3d_joints[:,:,:3].cpu().numpy(), reduction=None)
            else:
                error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices, has_smpl)
                error_joints = mean_per_joint_position_error(pred_3d_joints_from_smpl, gt_3d_joints,  has_3d_joints)
                error_joints_pa = reconstruction_error(pred_3d_joints_from_smpl.cpu().numpy(), gt_3d_joints[:,:,:3].cpu().numpy(), reduction=None)


            if len(error_vertices)>0:
                mPVE.update(np.mean(error_vertices), int(torch.sum(has_smpl)) )
            if len(error_joints)>0:
                mPJPE.update(np.mean(error_joints), int(torch.sum(has_3d_joints)) )
            if len(error_joints_pa)>0:
                PAmPJPE.update(np.mean(error_joints_pa), int(torch.sum(has_3d_joints)) )

    val_mPVE = all_gather(float(mPVE.avg))
    val_mPVE = sum(val_mPVE)/len(val_mPVE)
    val_mPJPE = all_gather(float(mPJPE.avg))
    val_mPJPE = sum(val_mPJPE)/len(val_mPJPE)

    val_PAmPJPE = all_gather(float(PAmPJPE.avg))
    val_PAmPJPE = sum(val_PAmPJPE)/len(val_PAmPJPE)

    val_count = all_gather(float(mPVE.count))
    val_count = sum(val_count)

    return val_mPVE, val_mPJPE, val_PAmPJPE, val_count


def visualize_mesh( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d,
                    mesh_2d):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    mesh_2d = mesh_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        mesh_2d_ = mesh_2d[i]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer, mesh_2d_)
        # rend_img = rend_img.transpose(2,0,1)
        rend_img = np.asarray(rend_img[:,:,::-1]*255)
        rend_imgs.append(rend_img)
    rend_imgs = np.vstack(rend_imgs)
    return rend_imgs

def visualize_mesh_test( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d,
                    PAmPJPE_h36m_j14):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        score = PAmPJPE_h36m_j14[i]
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction_test(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer, score)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_yaml", default='imagenet2012/train.yaml', type=str, required=False,
                        help="Yaml file with all data for training.")
    parser.add_argument("--val_yaml", default='imagenet2012/test.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.") 
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='tore/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=200, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=100.0, type=float)          
    parser.add_argument("--joints_loss_weight", default=1000.0, type=float)
    parser.add_argument("--vloss_w_full", default=0.33, type=float) 
    parser.add_argument("--vloss_w_sub", default=0.33, type=float) 
    parser.add_argument("--vloss_w_sub2", default=0.33, type=float) 
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument("--legacy_setting", default=True, action='store_true',)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
    parser.add_argument('--logging_steps', type=int, default=500, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")


    ##############################################
    #FAST
    parser.add_argument("--model_name", default='FastMETRO_L', type=str)
    parser.add_argument("--model_dim_1", default=512, type=int)
    parser.add_argument("--model_dim_2", default=128, type=int)
    parser.add_argument("--model_dim_3", default=128, type=int)
    parser.add_argument("--feedforward_dim_1", default=2048, type=int)
    parser.add_argument("--feedforward_dim_2", default=512, type=int)
    parser.add_argument("--feedforward_dim_3", default=512, type=int)
    parser.add_argument("--conv_1x1_dim", default=2048, type=int)
    parser.add_argument("--transformer_dropout", default=0.1, type=float)
    parser.add_argument("--transformer_nhead", default=8, type=int)
    parser.add_argument("--pos_type", default='sine', type=str)
    ###############
    parser.add_argument("--keep_ratio", default=None, type=float)
    parser.add_argument("--edge_and_normal_vector_loss", default=None, type=str)

    ###############
    #Eval
    parser.add_argument("--oracle", default="false", type=str)
    
    parser.add_argument("--itp_loss_weight", default=1e-3, type=float)


    
    args = parser.parse_args()
    return args


def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    if args.distributed:
        # print("Init distributed training on local rank {} ({}), rank {}, world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), int(os.environ["NODE_RANK"]), args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        synchronize()

    mkdir(args.output_dir)
    logger = setup_logger("METRO", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    renderer = Renderer(faces=smpl.faces.cpu().numpy())
    
    if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        elif 'efficientnet' in args.arch:
            # EfficientNet b0-b4
            print(f"loading pretrained {args.arch}")
            backbone = EfficientNet.from_pretrained(args.arch)
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        _metro_network = METRO_Network(args, backbone, mesh_sampler)

        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _metro_network.load_state_dict(state_dict, strict=False)
        del state_dict
    else:
        # init ImageNet pre-trained backbone model
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        elif 'efficientnet' in args.arch:
            # Peter: EfficientNet b0-b4
            print(f"loading pretrained {args.arch}")
            backbone = EfficientNet.from_pretrained(args.arch)
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        _metro_network = METRO_Network(args, backbone, mesh_sampler)

        print(args.keep_ratio)
        total_params = sum([params.numel() for name, params in _metro_network.named_parameters()])
        backbone_params = sum([params.numel() for name, params in _metro_network.named_parameters() if 'backbone' in name])
        transformer_params = sum([params.numel() for name, params in _metro_network.named_parameters() if 'transformer_1' in name or 'transformer_2' in name])
        nsr_params = sum([params.numel() for name, params in _metro_network.named_parameters() if 'transformer_3' in name])
        others_params = sum([params.numel() for name, params in _metro_network.named_parameters() if 'backbone' not in name and 'transformer_1' not in name and 'transformer_2' not in name and 'transformer_3' not in name])
        print(f"total params: {total_params}")
        print(f"backbone params: {backbone_params}")
        print(f"transformer params: {transformer_params}")
        print(f"NSR params: {nsr_params}")
        print(f"others params: {others_params}")

        if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            cpu_device = torch.device('cpu')
            state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
            _metro_network.load_state_dict(state_dict, strict=False)
            del state_dict
    
    _metro_network.to(args.device)
    logger.info("Training parameters %s", args)

    if args.run_eval_only==True:
        val_dataloader = make_data_loader(args, args.val_yaml, 
                                        args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        run_eval_general(args, val_dataloader, _metro_network, smpl, mesh_sampler)

    else:
        train_dataloader = make_data_loader(args, args.train_yaml, 
                                            args.distributed, is_train=True, scale_factor=args.img_scale_factor)
        val_dataloader = make_data_loader(args, args.val_yaml, 
                                        args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        run(args, train_dataloader, val_dataloader, _metro_network, smpl, mesh_sampler, renderer)



if __name__ == "__main__":
    args = parse_args()
    main(args)
