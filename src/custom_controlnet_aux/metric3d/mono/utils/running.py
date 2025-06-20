import os
import torch
import torch.nn as nn
from .comm import main_process
import copy
import inspect
import logging
import glob


def load_ckpt(load_path, model, optimizer=None, scheduler=None, strict_match=True, loss_scaler=None):
    """
    Load the check point for resuming training or finetuning.
    """
    logger = logging.getLogger()
    if os.path.isfile(load_path):
        if main_process():
            logger.info(f"Loading weight '{load_path}'")
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)
        ckpt_state_dict  = checkpoint['model_state_dict']
        model.load_state_dict(ckpt_state_dict, strict=strict_match)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if loss_scaler is not None and 'scaler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scaler'])
        del ckpt_state_dict
        del checkpoint
        if main_process():
            logger.info(f"Successfully loaded weight: '{load_path}'")
            if scheduler is not None and optimizer is not None:
                logger.info(f"Resume training from: '{load_path}'")
    else:
        if main_process():
            raise RuntimeError(f"No weight found at '{load_path}'")
    return model, optimizer, scheduler, loss_scaler


def save_ckpt(cfg, model, optimizer, scheduler, curr_iter=0, curr_epoch=None, loss_scaler=None):
    """
    Save the model, optimizer, lr scheduler.
    """
    logger = logging.getLogger()

    if 'IterBasedRunner' in cfg.runner.type:
        max_iters = cfg.runner.max_iters
    elif 'EpochBasedRunner' in cfg.runner.type:
        max_iters = cfg.runner.max_epochs
    else:
        raise TypeError(f'{cfg.runner.type} is not supported')

    ckpt = dict(
        model_state_dict=model.module.state_dict(),
        optimizer=optimizer.state_dict(),
        max_iter=cfg.runner.max_iters if 'max_iters' in cfg.runner \
            else cfg.runner.max_epochs,
        scheduler=scheduler.state_dict(),
    )

    if loss_scaler is not None:
        ckpt.update(dict(scaler=loss_scaler.state_dict()))
    
    ckpt_dir = os.path.join(cfg.work_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    save_name = os.path.join(ckpt_dir, 'step%08d.pth' %curr_iter)
    saved_ckpts = glob.glob(ckpt_dir + '/step*.pth')
    torch.save(ckpt, save_name)

    # keep the last 8 ckpts
    if len(saved_ckpts) > 20:
        saved_ckpts.sort()
        os.remove(saved_ckpts.pop(0))
    
    logger.info(f'Save model: {save_name}')
