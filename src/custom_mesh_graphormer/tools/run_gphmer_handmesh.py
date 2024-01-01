"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training and evaluation codes for 
3D hand mesh reconstruction from an image
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
from custom_mesh_graphormer.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from custom_mesh_graphormer.modeling._mano import MANO, Mesh
from custom_mesh_graphormer.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from custom_mesh_graphormer.modeling.hrnet.config import config as hrnet_config
from custom_mesh_graphormer.modeling.hrnet.config import update_config as hrnet_update_config
import custom_mesh_graphormer.modeling.data.config as cfg
from custom_mesh_graphormer.datasets.build import make_hand_data_loader

from custom_mesh_graphormer.utils.logger import setup_logger
from custom_mesh_graphormer.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from custom_mesh_graphormer.utils.miscellaneous import mkdir, set_seed
from custom_mesh_graphormer.utils.metric_logger import AverageMeter
from custom_mesh_graphormer.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test, visualize_reconstruction_no_text
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

def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs/2.0)  ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence is binary and indicates whether the keypoints exist or not.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_root = gt_keypoints_3d[:, 0,:]
        gt_keypoints_3d = gt_keypoints_3d - gt_root[:, None, :]
        pred_root = pred_keypoints_3d[:, 0,:]
        pred_keypoints_3d = pred_keypoints_3d - pred_root[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)
    

def run(args, train_dataloader, Graphormer_model, mano_model, renderer, mesh_sampler):

    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs

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

    start_training_time = time.time()
    end = time.time()
    Graphormer_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_loss_2djoints = AverageMeter()
    log_loss_3djoints = AverageMeter()
    log_loss_vertices = AverageMeter()

    for iteration, (img_keys, images, annotations) in enumerate(train_dataloader):

        Graphormer_model.train()
        iteration += 1
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)
        adjust_learning_rate(optimizer, epoch, args)
        data_time.update(time.time() - end)

        images = images.to(device)
        gt_2d_joints = annotations['joints_2d'].to(device)
        gt_pose = annotations['pose'].to(device)
        gt_betas = annotations['betas'].to(device)
        has_mesh = annotations['has_smpl'].to(device)
        has_3d_joints = has_mesh
        has_2d_joints = has_mesh
        mjm_mask = annotations['mjm_mask'].to(device)
        mvm_mask = annotations['mvm_mask'].to(device)

        # generate mesh
        gt_vertices, gt_3d_joints = mano_model.layer(gt_pose, gt_betas)
        gt_vertices = gt_vertices/1000.0
        gt_3d_joints = gt_3d_joints/1000.0

        gt_vertices_sub = mesh_sampler.downsample(gt_vertices)
        # normalize gt based on hand's wrist 
        gt_3d_root = gt_3d_joints[:,cfg.J_NAME.index('Wrist'),:]
        gt_vertices = gt_vertices - gt_3d_root[:, None, :]
        gt_vertices_sub = gt_vertices_sub - gt_3d_root[:, None, :]
        gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
        gt_3d_joints_with_tag = torch.ones((batch_size,gt_3d_joints.shape[1],4)).to(device)
        gt_3d_joints_with_tag[:,:,:3] = gt_3d_joints

        # prepare masks for mask vertex/joint modeling
        mjm_mask_ = mjm_mask.expand(-1,-1,2051)
        mvm_mask_ = mvm_mask.expand(-1,-1,2051)
        meta_masks = torch.cat([mjm_mask_, mvm_mask_], dim=1)
        
        # forward-pass
        pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices = Graphormer_model(images, mano_model, mesh_sampler, meta_masks=meta_masks, is_train=True)

        # obtain 3d joints, which are regressed from the full mesh
        pred_3d_joints_from_mesh = mano_model.get_3d_joints(pred_vertices)

        # obtain 2d joints, which are projected from 3d joints of smpl mesh
        pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
        pred_2d_joints = orthographic_projection(pred_3d_joints.contiguous(), pred_camera.contiguous())
        
        # compute 3d joint loss  (where the joints are directly output from transformer)
        loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints_with_tag, has_3d_joints)

        # compute 3d vertex loss
        loss_vertices = ( args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub, has_mesh) + \
                            args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_mesh) )

        # compute 3d joint loss (where the joints are regressed from full mesh)
        loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_mesh, gt_3d_joints_with_tag, has_3d_joints)
        # compute 2d joint loss
        loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints, has_2d_joints)  + \
                         keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_mesh, gt_2d_joints, has_2d_joints)
        
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
                                            pred_2d_joints_from_mesh.detach())
            visual_imgs = visual_imgs.transpose(0,1)
            visual_imgs = visual_imgs.transpose(1,2)
            visual_imgs = np.asarray(visual_imgs)

            if is_main_process()==True:
                stamp = str(epoch) + '_' + str(iteration)
                temp_fname = args.output_dir + 'visual_' + stamp + '.jpg'
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))
                aml_run.log_image(name='visual results', path=temp_fname)

        if iteration % iters_per_epoch == 0:
            if epoch%10==0:
                checkpoint_dir = save_checkpoint(Graphormer_model, args, epoch, iteration)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    checkpoint_dir = save_checkpoint(Graphormer_model, args, epoch, iteration)

def run_eval_and_save(args, split, val_dataloader, Graphormer_model, mano_model, renderer, mesh_sampler):

    criterion_keypoints = torch.nn.MSELoss(reduction='none').to(device)
    criterion_vertices = torch.nn.L1Loss().to(device)

    if args.distributed:
        Graphormer_model = torch.nn.parallel.DistributedDataParallel(
            Graphormer_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    Graphormer_model.eval()

    if args.aml_eval==True:
        run_aml_inference_hand_mesh(args, val_dataloader, 
                                Graphormer_model, 
                                criterion_keypoints, 
                                criterion_vertices, 
                                0, 
                                mano_model, mesh_sampler,
                                renderer, split)
    else:
        run_inference_hand_mesh(args, val_dataloader, 
                                Graphormer_model, 
                                criterion_keypoints, 
                                criterion_vertices, 
                                0, 
                                mano_model, mesh_sampler,
                                renderer, split)
    checkpoint_dir = save_checkpoint(Graphormer_model, args, 0, 0)
    return

def run_aml_inference_hand_mesh(args, val_loader, Graphormer_model, criterion, criterion_vertices, epoch, mano_model, mesh_sampler, renderer, split):
    # switch to evaluate mode
    Graphormer_model.eval()
    fname_output_save = []
    mesh_output_save = []
    joint_output_save = []
    world_size = get_world_size()
    with torch.no_grad():
        for i, (img_keys, images, annotations) in enumerate(val_loader):
            batch_size = images.size(0)
            # compute output
            images = images.to(device)
            
            # forward-pass
            pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices = Graphormer_model(images, mano_model, mesh_sampler)
            # obtain 3d joints from full mesh
            pred_3d_joints_from_mesh = mano_model.get_3d_joints(pred_vertices)

            for j in range(batch_size):
                fname_output_save.append(img_keys[j])
                pred_vertices_list = pred_vertices[j].tolist()
                mesh_output_save.append(pred_vertices_list)
                pred_3d_joints_from_mesh_list = pred_3d_joints_from_mesh[j].tolist()
                joint_output_save.append(pred_3d_joints_from_mesh_list)

    if world_size > 1:
        torch.distributed.barrier()
    print('save results to pred.json')
    output_json_file = 'pred.json'
    print('save results to ', output_json_file)
    with open(output_json_file, 'w') as f:
        json.dump([joint_output_save, mesh_output_save], f)

    azure_ckpt_name = '200' # args.resume_checkpoint.split('/')[-2].split('-')[1]
    inference_setting = 'sc%02d_rot%s'%(int(args.sc*10),str(int(args.rot)))
    output_zip_file = args.output_dir + 'ckpt' + azure_ckpt_name + '-' + inference_setting +'-pred.zip'

    resolved_submit_cmd = 'zip ' + output_zip_file + ' ' + output_json_file
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    resolved_submit_cmd = 'rm %s'%(output_json_file)
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    if world_size > 1:
        torch.distributed.barrier()

    return 

def run_inference_hand_mesh(args, val_loader, Graphormer_model, criterion, criterion_vertices, epoch, mano_model, mesh_sampler, renderer, split):
    # switch to evaluate mode
    Graphormer_model.eval()
    fname_output_save = []
    mesh_output_save = []
    joint_output_save = []
    with torch.no_grad():
        for i, (img_keys, images, annotations) in enumerate(val_loader):
            batch_size = images.size(0)
            # compute output
            images = images.to(device)

            # forward-pass
            pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices = Graphormer_model(images, mano_model, mesh_sampler)

            # obtain 3d joints from full mesh
            pred_3d_joints_from_mesh = mano_model.get_3d_joints(pred_vertices)
            pred_3d_pelvis = pred_3d_joints_from_mesh[:,cfg.J_NAME.index('Wrist'),:]
            pred_3d_joints_from_mesh = pred_3d_joints_from_mesh - pred_3d_pelvis[:, None, :]
            pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

            for j in range(batch_size):
                fname_output_save.append(img_keys[j])
                pred_vertices_list = pred_vertices[j].tolist()
                mesh_output_save.append(pred_vertices_list)
                pred_3d_joints_from_mesh_list = pred_3d_joints_from_mesh[j].tolist()
                joint_output_save.append(pred_3d_joints_from_mesh_list)

            if i%20==0:
                # obtain 3d joints, which are regressed from the full mesh
                pred_3d_joints_from_mesh = mano_model.get_3d_joints(pred_vertices)
                # obtain 2d joints, which are projected from 3d joints of mesh
                pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
                visual_imgs = visualize_mesh(   renderer,
                                                annotations['ori_img'].detach(),
                                                annotations['joints_2d'].detach(),
                                                pred_vertices.detach(), 
                                                pred_camera.detach(),
                                                pred_2d_joints_from_mesh.detach())

                visual_imgs = visual_imgs.transpose(0,1)
                visual_imgs = visual_imgs.transpose(1,2)
                visual_imgs = np.asarray(visual_imgs)
                
                inference_setting = 'sc%02d_rot%s'%(int(args.sc*10),str(int(args.rot)))
                temp_fname = args.output_dir + args.resume_checkpoint[0:-9] + 'freihand_results_'+inference_setting+'_batch'+str(i)+'.jpg'
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

    print('save results to pred.json')
    with open('pred.json', 'w') as f:
        json.dump([joint_output_save, mesh_output_save], f)

    run_exp_name = args.resume_checkpoint.split('/')[-3]
    run_ckpt_name = args.resume_checkpoint.split('/')[-2].split('-')[1]
    inference_setting = 'sc%02d_rot%s'%(int(args.sc*10),str(int(args.rot)))
    resolved_submit_cmd = 'zip ' + args.output_dir + run_exp_name + '-ckpt'+ run_ckpt_name + '-' + inference_setting +'-pred.zip  ' +  'pred.json'
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    resolved_submit_cmd = 'rm pred.json'
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    return 

def visualize_mesh( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(21))
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
                    PAmPJPE):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(21))
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
        score = PAmPJPE[i]
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction_test(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer, score)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

def visualize_mesh_no_text( renderer,
                    images,
                    pred_vertices, 
                    pred_camera):
    """Tensorboard logging."""
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 1)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction only
        rend_img = visualize_reconstruction_no_text(img, 224, vertices, cam, renderer, color='hand')
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
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=200, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=1.0, type=float)          
    parser.add_argument("--joints_loss_weight", default=1.0, type=float)
    parser.add_argument("--vloss_w_full", default=0.5, type=float) 
    parser.add_argument("--vloss_w_sub", default=0.5, type=float)  
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument("--num_hidden_layers", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=-1, type=int, required=False, 
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
    parser.add_argument("--mesh_type", default='hand', type=str, help="body or hand") 

    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
    parser.add_argument("--multiscale_inference", default=False, action='store_true',) 
    # if enable "multiscale_inference", dataloader will apply transformations to the test image based on
    # the rotation "rot" and scale "sc" parameters below 
    parser.add_argument("--rot", default=0, type=float) 
    parser.add_argument("--sc", default=1.0, type=float) 
    parser.add_argument("--aml_eval", default=False, action='store_true',) 

    parser.add_argument('--logging_steps', type=int, default=100, 
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
        print("Init distributed training on local rank {}".format(args.local_rank))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        synchronize()
   
    mkdir(args.output_dir)
    logger = setup_logger("Graphormer", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.to(device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    renderer = Renderer(faces=mano_model.face)

    # Load pretrained model
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
            args.intermediate_size = int(args.hidden_size*2)

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
        
        # create backbone model
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
            backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Graphormer encoders total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
        _model = Graphormer_Network(args, config, backbone, trans_encoder)

        if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            # workaround approach to load sparse tensor in graph conv.
            state_dict = torch.load(args.resume_checkpoint)
            _model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
    
    _model.to(args.device)
    logger.info("Training parameters %s", args)

    if args.run_eval_only==True:
        val_dataloader = make_hand_data_loader(args, args.val_yaml, 
                                        args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        run_eval_and_save(args, 'freihand', val_dataloader, _model, mano_model, renderer, mesh_sampler)

    else:
        train_dataloader = make_hand_data_loader(args, args.train_yaml, 
                                            args.distributed, is_train=True, scale_factor=args.img_scale_factor)
        run(args, train_dataloader, _model, mano_model, renderer, mesh_sampler)

if __name__ == "__main__":
    args = parse_args()
    main(args)
