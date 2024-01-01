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
import gc
import numpy as np
import cv2
from custom_mesh_graphormer.modeling.bert import BertConfig, Graphormer
from custom_mesh_graphormer.modeling.bert import Graphormer_Body_Network as Graphormer_Network
from custom_mesh_graphormer.modeling._smpl import SMPL, Mesh
from custom_mesh_graphormer.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from custom_mesh_graphormer.modeling.hrnet.config import config as hrnet_config
from custom_mesh_graphormer.modeling.hrnet.config import update_config as hrnet_update_config
import custom_mesh_graphormer.modeling.data.config as cfg
from custom_mesh_graphormer.datasets.build import make_data_loader

from custom_mesh_graphormer.utils.logger import setup_logger
from custom_mesh_graphormer.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from custom_mesh_graphormer.utils.miscellaneous import mkdir, set_seed
from custom_mesh_graphormer.utils.metric_logger import AverageMeter, EvalMetricsLogger
from custom_mesh_graphormer.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test
from custom_mesh_graphormer.utils.metric_pampjpe import reconstruction_error
from custom_mesh_graphormer.utils.geometric_layers import orthographic_projection

from comfy.model_management import get_torch_device
device = get_torch_device()

from azureml.core.run import Run
aml_run = Run.get_context()

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

def run(args, train_dataloader, val_dataloader, Graphormer_model, smpl, mesh_sampler, renderer):
    smpl.eval()
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs
    if iters_per_epoch<1000:
        args.logging_steps = 500

    optimizer = torch.optim.Adam(params=list(Graphormer_model.parameters()),
                                           lr=args.lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)

    # define loss function (criterion) and optimizer
    criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').to(device)
    criterion_keypoints = torch.nn.MSELoss(reduction='none').to(device)
    criterion_vertices = torch.nn.L1Loss().to(device)

    if args.distributed:
        Graphormer_model = torch.nn.parallel.DistributedDataParallel(
            Graphormer_model, device_ids=[args.local_rank], 
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
    Graphormer_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_loss_2djoints = AverageMeter()
    log_loss_3djoints = AverageMeter()
    log_loss_vertices = AverageMeter()
    log_eval_metrics = EvalMetricsLogger()

    for iteration, (img_keys, images, annotations) in enumerate(train_dataloader):
        # gc.collect()
        # torch.cuda.empty_cache()
        Graphormer_model.train()
        iteration += 1
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)
        adjust_learning_rate(optimizer, epoch, args)
        data_time.update(time.time() - end)

        images = images.to(device)
        gt_2d_joints = annotations['joints_2d'].to(device)
        gt_2d_joints = gt_2d_joints[:,cfg.J24_TO_J14,:]
        has_2d_joints = annotations['has_2d_joints'].to(device)

        gt_3d_joints = annotations['joints_3d'].to(device)
        gt_3d_pelvis = gt_3d_joints[:,cfg.J24_NAME.index('Pelvis'),:3]
        gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:] 
        gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :]
        has_3d_joints = annotations['has_3d_joints'].to(device)

        gt_pose = annotations['pose'].to(device)
        gt_betas = annotations['betas'].to(device)
        has_smpl = annotations['has_smpl'].to(device)
        mjm_mask = annotations['mjm_mask'].to(device)
        mvm_mask = annotations['mvm_mask'].to(device)

        # generate simplified mesh
        gt_vertices = smpl(gt_pose, gt_betas)
        gt_vertices_sub2 = mesh_sampler.downsample(gt_vertices, n1=0, n2=2)
        gt_vertices_sub = mesh_sampler.downsample(gt_vertices)

        # normalize gt based on smpl's pelvis 
        gt_smpl_3d_joints = smpl.get_h36m_joints(gt_vertices)
        gt_smpl_3d_pelvis = gt_smpl_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
        gt_vertices_sub2 = gt_vertices_sub2 - gt_smpl_3d_pelvis[:, None, :]
            
        # prepare masks for mask vertex/joint modeling
        mjm_mask_ = mjm_mask.expand(-1,-1,2051)
        mvm_mask_ = mvm_mask.expand(-1,-1,2051)
        meta_masks = torch.cat([mjm_mask_, mvm_mask_], dim=1)

        # forward-pass
        pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = Graphormer_model(images, smpl, mesh_sampler, meta_masks=meta_masks, is_train=True)

        # normalize gt based on smpl's pelvis 
        gt_vertices_sub = gt_vertices_sub - gt_smpl_3d_pelvis[:, None, :] 
        gt_vertices = gt_vertices - gt_smpl_3d_pelvis[:, None, :]

        # obtain 3d joints, which are regressed from the full mesh
        pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
        pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]

        # obtain 2d joints, which are projected from 3d joints of smpl mesh
        pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
        pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

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
    
        # we empirically use hyperparameters to balance difference losses
        loss = args.joints_loss_weight*loss_3d_joints + \
                args.vertices_loss_weight*loss_vertices  + args.vertices_loss_weight*loss_2d_joints

        # update logs
        log_loss_2djoints.update(loss_2d_joints.item(), batch_size)
        log_loss_3djoints.update(loss_3d_joints.item(), batch_size)
        log_loss_vertices.update(loss_vertices.item(), batch_size)
        log_losses.update(loss.item(), batch_size)

        # back prop
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % args.logging_steps == 0 or iteration == max_iter:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                ' '.join(
                ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}',]
                ).format(eta=eta_string, ep=epoch, iter=iteration, 
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) 
                + '  loss: {:.4f}, 2d joint loss: {:.4f}, 3d joint loss: {:.4f}, vertex loss: {:.4f}, compute: {:.4f}, data: {:.4f}, lr: {:.6f}'.format(
                    log_losses.avg, log_loss_2djoints.avg, log_loss_3djoints.avg, log_loss_vertices.avg, batch_time.avg, data_time.avg, 
                    optimizer.param_groups[0]['lr'])
            )

            aml_run.log(name='Loss', value=float(log_losses.avg))
            aml_run.log(name='3d joint Loss', value=float(log_loss_3djoints.avg))
            aml_run.log(name='2d joint Loss', value=float(log_loss_2djoints.avg))
            aml_run.log(name='vertex Loss', value=float(log_loss_vertices.avg))

            visual_imgs = visualize_mesh(   renderer,
                                            annotations['ori_img'].detach(),
                                            annotations['joints_2d'].detach(),
                                            pred_vertices.detach(), 
                                            pred_camera.detach(),
                                            pred_2d_joints_from_smpl.detach())
            visual_imgs = visual_imgs.transpose(0,1)
            visual_imgs = visual_imgs.transpose(1,2)
            visual_imgs = np.asarray(visual_imgs)

            if is_main_process()==True:
                stamp = str(epoch) + '_' + str(iteration)
                temp_fname = args.output_dir + 'visual_' + stamp + '.jpg'
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))
                aml_run.log_image(name='visual results', path=temp_fname)

        if iteration % iters_per_epoch == 0:
            val_mPVE, val_mPJPE, val_PAmPJPE, val_count = run_validate(args, val_dataloader, 
                                                Graphormer_model, 
                                                criterion_keypoints, 
                                                criterion_vertices, 
                                                epoch, 
                                                smpl,
                                                mesh_sampler)
            aml_run.log(name='mPVE', value=float(1000*val_mPVE))
            aml_run.log(name='mPJPE', value=float(1000*val_mPJPE))
            aml_run.log(name='PAmPJPE', value=float(1000*val_PAmPJPE))
            logger.info(
                ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
                + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}, Data Count: {:6.2f}'.format(1000*val_mPVE, 1000*val_mPJPE, 1000*val_PAmPJPE, val_count)
            )

            if val_PAmPJPE<log_eval_metrics.PAmPJPE:
                checkpoint_dir = save_checkpoint(Graphormer_model, args, epoch, iteration)
                log_eval_metrics.update(val_mPVE, val_mPJPE, val_PAmPJPE, epoch)
                
        
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    checkpoint_dir = save_checkpoint(Graphormer_model, args, epoch, iteration)

    logger.info(
        ' Best Results:'
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}, at epoch {:6.2f}'.format(1000*log_eval_metrics.mPVE, 1000*log_eval_metrics.mPJPE, 1000*log_eval_metrics.PAmPJPE, log_eval_metrics.epoch)
    )


def run_eval_general(args, val_dataloader, Graphormer_model, smpl, mesh_sampler):
    smpl.eval()
    criterion_keypoints = torch.nn.MSELoss(reduction='none').to(device)
    criterion_vertices = torch.nn.L1Loss().to(device)

    epoch = 0
    if args.distributed:
        Graphormer_model = torch.nn.parallel.DistributedDataParallel(
            Graphormer_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    Graphormer_model.eval()

    val_mPVE, val_mPJPE, val_PAmPJPE, val_count = run_validate(args, val_dataloader, 
                                    Graphormer_model, 
                                    criterion_keypoints, 
                                    criterion_vertices, 
                                    epoch, 
                                    smpl,
                                    mesh_sampler)

    aml_run.log(name='mPVE', value=float(1000*val_mPVE))
    aml_run.log(name='mPJPE', value=float(1000*val_mPJPE))
    aml_run.log(name='PAmPJPE', value=float(1000*val_PAmPJPE))

    logger.info(
        ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f} '.format(1000*val_mPVE, 1000*val_mPJPE, 1000*val_PAmPJPE)
    )
    # checkpoint_dir = save_checkpoint(Graphormer_model, args, 0, 0)
    return

def run_validate(args, val_loader, Graphormer_model, criterion, criterion_vertices, epoch, smpl, mesh_sampler):
    batch_time = AverageMeter()
    mPVE = AverageMeter()
    mPJPE = AverageMeter()
    PAmPJPE = AverageMeter()
    # switch to evaluate mode
    Graphormer_model.eval()
    smpl.eval()
    with torch.no_grad():
        # end = time.time()
        for i, (img_keys, images, annotations) in enumerate(val_loader):
            batch_size = images.size(0)
            # compute output
            images = images.to(device)
            gt_3d_joints = annotations['joints_3d'].to(device)
            gt_3d_pelvis = gt_3d_joints[:,cfg.J24_NAME.index('Pelvis'),:3]
            gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:] 
            gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :]
            has_3d_joints = annotations['has_3d_joints'].to(device)

            gt_pose = annotations['pose'].to(device)
            gt_betas = annotations['betas'].to(device)
            has_smpl = annotations['has_smpl'].to(device)

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
            pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = Graphormer_model(images, smpl, mesh_sampler)

            # obtain 3d joints from full mesh
            pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

            pred_3d_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
            pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

            # measure errors
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
                    pred_keypoints_2d):
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
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
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
    parser.add_argument("--model_name_or_path", default='src/modeling/bert/bert-base-uncased/', type=str, required=False,
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
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument("--which_gcn", default='0,0,1', type=str, 
                        help="which encoder block to have graph conv. Encoder1, Encoder2, Encoder3. Default: only Encoder3 has graph conv") 
    parser.add_argument("--mesh_type", default='body', type=str, help="body or hand") 
    parser.add_argument("--interm_size_scale", default=2, type=int)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
    parser.add_argument('--logging_steps', type=int, default=1000, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")


    args = parser.parse_args()
    return args


def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))

    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    if args.distributed:
        print("Init distributed training on local rank {} ({}), rank {}, world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), int(os.environ["NODE_RANK"]), args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        synchronize()

    mkdir(args.output_dir)
    logger = setup_logger("Graphormer", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    renderer = Renderer(faces=smpl.faces.cpu().numpy())

    # Load model
    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]

    # which encoder block to have graph convs
    which_blk_graph = [int(item) for item in args.which_gcn.split(',')]
    
    if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _model = torch.load(args.resume_checkpoint)
    else:
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            config = config_class.from_pretrained(args.config_name if args.config_name \
                    else args.model_name_or_path)

            config.output_attentions = False
            config.hidden_dropout_prob = args.drop_out
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size*args.interm_size_scale)

            if which_blk_graph[i]==1:
                config.graph_conv = True
                logger.info("Add Graph Conv")
            else:
                config.graph_conv = False

            config.mesh_type = args.mesh_type

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            trans_encoder.append(model)

        
        # init ImageNet pre-trained backbone model
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])


        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Graphormer encoders total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end Graphormer network (CNN backbone + multi-layer graphormer encoder)
        _model = Graphormer_Network(args, config, backbone, trans_encoder, mesh_sampler)

        if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            # workaround approach to load sparse tensor in graph conv.
            states = torch.load(args.resume_checkpoint)
            # states = checkpoint_loaded.state_dict()
            for k, v in states.items():
                states[k] = v.cpu()
            # del checkpoint_loaded
            _model.load_state_dict(states, strict=False)
            del states
            gc.collect()
            torch.cuda.empty_cache()


    _model.to(args.device)
    logger.info("Training parameters %s", args)

    if args.run_eval_only==True:
        val_dataloader = make_data_loader(args, args.val_yaml, 
                                        args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        run_eval_general(args, val_dataloader, _model, smpl, mesh_sampler)

    else:
        train_dataloader = make_data_loader(args, args.train_yaml, 
                                            args.distributed, is_train=True, scale_factor=args.img_scale_factor)
        val_dataloader = make_data_loader(args, args.val_yaml, 
                                        args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        run(args, train_dataloader, val_dataloader, _model, smpl, mesh_sampler, renderer)



if __name__ == "__main__":
    args = parse_args()
    main(args)
