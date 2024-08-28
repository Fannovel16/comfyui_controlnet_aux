import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = np.longdouble(0.0)
        self.avg = np.longdouble(0.0)
        self.sum = np.longdouble(0.0)
        self.count = np.longdouble(0.0)

    def update(self, val, n: float = 1) -> None:
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / (self.count + 1e-6)

class MetricAverageMeter(AverageMeter):
    """ 
    An AverageMeter designed specifically for evaluating segmentation results.
    """
    def __init__(self, metrics: list) -> None:
        """ Initialize object. """
        # average meters for metrics
        self.abs_rel = AverageMeter()
        self.rmse = AverageMeter()
        self.silog = AverageMeter()
        self.delta1 = AverageMeter()
        self.delta2 = AverageMeter()
        self.delta3 = AverageMeter()

        self.metrics = metrics

        self.consistency = AverageMeter()
        self.log10 = AverageMeter()
        self.rmse_log = AverageMeter()
        self.sq_rel = AverageMeter()

        # normal
        self.normal_mean = AverageMeter()
        self.normal_rmse = AverageMeter()
        self.normal_a1 = AverageMeter()
        self.normal_a2 = AverageMeter()
        
        self.normal_median = AverageMeter()
        self.normal_a3 = AverageMeter()
        self.normal_a4 = AverageMeter()
        self.normal_a5 = AverageMeter()


    def update_metrics_cpu(self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,):
        """
        Update metrics on cpu
        """

        assert pred.shape == target.shape

        if len(pred.shape) == 3:
            pred = pred[:, None, :, :]
            target = target[:, None, :, :]
            mask = mask[:, None, :, :]
        elif len(pred.shape) == 2:
            pred = pred[None, None, :, :]
            target = target[None, None, :, :]
            mask = mask[None, None, :, :]


        # Absolute relative error
        abs_rel_sum, valid_pics = get_absrel_err(pred, target, mask)
        abs_rel_sum = abs_rel_sum.numpy()
        valid_pics = valid_pics.numpy()
        self.abs_rel.update(abs_rel_sum, valid_pics)
        
        # squared relative error
        sqrel_sum, _ = get_sqrel_err(pred, target, mask)
        sqrel_sum = sqrel_sum.numpy()
        self.sq_rel.update(sqrel_sum, valid_pics)

        # root mean squared error
        rmse_sum, _ = get_rmse_err(pred, target, mask)
        rmse_sum = rmse_sum.numpy()
        self.rmse.update(rmse_sum, valid_pics)
        
        # log root mean squared error
        log_rmse_sum, _ = get_rmse_log_err(pred, target, mask)
        log_rmse_sum = log_rmse_sum.numpy()
        self.rmse.update(log_rmse_sum, valid_pics)
        
        # log10 error
        log10_sum, _ = get_log10_err(pred, target, mask)
        log10_sum = log10_sum.numpy()
        self.rmse.update(log10_sum, valid_pics)

        # scale-invariant root mean squared error in log space
        silog_sum, _ = get_silog_err(pred, target, mask)
        silog_sum = silog_sum.numpy()
        self.silog.update(silog_sum, valid_pics)

        # ratio error, delta1, ....
        delta1_sum, delta2_sum, delta3_sum, _ = get_ratio_error(pred, target, mask)
        delta1_sum = delta1_sum.numpy()
        delta2_sum = delta2_sum.numpy()
        delta3_sum = delta3_sum.numpy()

        self.delta1.update(delta1_sum, valid_pics)
        self.delta2.update(delta1_sum, valid_pics)
        self.delta3.update(delta1_sum, valid_pics)
        

    def update_metrics_gpu(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        is_distributed: bool,
        pred_next: torch.tensor = None,
        pose_f1_to_f2: torch.tensor = None,
        intrinsic: torch.tensor = None):
        """ 
        Update metric on GPU. It supports distributed processing. If multiple machines are employed, please
        set 'is_distributed' as True.
        """
        assert pred.shape == target.shape

        if len(pred.shape) == 3:
            pred = pred[:, None, :, :]
            target = target[:, None, :, :]
            mask = mask[:, None, :, :]
        elif len(pred.shape) == 2:
            pred = pred[None, None, :, :]
            target = target[None, None, :, :]
            mask = mask[None, None, :, :]


        # Absolute relative error
        abs_rel_sum, valid_pics = get_absrel_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(abs_rel_sum), dist.all_reduce(valid_pics)
        abs_rel_sum = abs_rel_sum.cpu().numpy()
        valid_pics = int(valid_pics)
        self.abs_rel.update(abs_rel_sum, valid_pics)

        # root mean squared error
        rmse_sum, _ = get_rmse_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(rmse_sum)
        rmse_sum = rmse_sum.cpu().numpy()
        self.rmse.update(rmse_sum, valid_pics)
        
        # log root mean squared error
        log_rmse_sum, _ = get_rmse_log_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(log_rmse_sum)
        log_rmse_sum = log_rmse_sum.cpu().numpy()
        self.rmse_log.update(log_rmse_sum, valid_pics)
    
        # log10 error
        log10_sum, _ = get_log10_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(log10_sum)
        log10_sum = log10_sum.cpu().numpy()
        self.log10.update(log10_sum, valid_pics)

        # scale-invariant root mean squared error in log space
        silog_sum, _ = get_silog_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(silog_sum)
        silog_sum = silog_sum.cpu().numpy()
        self.silog.update(silog_sum, valid_pics)

        # ratio error, delta1, ....
        delta1_sum, delta2_sum, delta3_sum, _ = get_ratio_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(delta1_sum), dist.all_reduce(delta2_sum), dist.all_reduce(delta3_sum)
        delta1_sum = delta1_sum.cpu().numpy()
        delta2_sum = delta2_sum.cpu().numpy()
        delta3_sum = delta3_sum.cpu().numpy()

        self.delta1.update(delta1_sum, valid_pics)
        self.delta2.update(delta2_sum, valid_pics)
        self.delta3.update(delta3_sum, valid_pics)

        # video consistency error
        # consistency_rel_sum, valid_warps = get_video_consistency_err(pred, pred_next, pose_f1_to_f2, intrinsic)
        # if is_distributed:
        #     dist.all_reduce(consistency_rel_sum), dist.all_reduce(valid_warps)
        # consistency_rel_sum = consistency_rel_sum.cpu().numpy()
        # valid_warps = int(valid_warps)
        # self.consistency.update(consistency_rel_sum, valid_warps)

    ## for surface normal
    def update_normal_metrics_gpu(
        self,
        pred: torch.Tensor, # (B, 3, H, W)
        target: torch.Tensor, # (B, 3, H, W)
        mask: torch.Tensor, # (B, 1, H, W)
        is_distributed: bool,
        ):
        """ 
        Update metric on GPU. It supports distributed processing. If multiple machines are employed, please
        set 'is_distributed' as True.
        """
        assert pred.shape == target.shape

        valid_pics = torch.sum(mask, dtype=torch.float32) + 1e-6

        if valid_pics < 10:
            return

        mean_error = rmse_error = a1_error = a2_error = dist_node_cnt = valid_pics
        normal_error = torch.cosine_similarity(pred, target, dim=1)
        normal_error = torch.clamp(normal_error, min=-1.0, max=1.0)
        angle_error = torch.acos(normal_error) * 180.0 / torch.pi
        angle_error = angle_error[:, None, :, :]
        angle_error = angle_error[mask]
        # Calculation error
        mean_error = angle_error.sum() / valid_pics
        rmse_error = torch.sqrt( torch.sum(torch.square(angle_error)) / valid_pics )
        median_error = angle_error.median()
        a1_error = 100.0 * (torch.sum(angle_error < 5) / valid_pics)
        a2_error = 100.0 * (torch.sum(angle_error < 7.5) / valid_pics)
        
        a3_error = 100.0 * (torch.sum(angle_error < 11.25) / valid_pics)
        a4_error = 100.0 * (torch.sum(angle_error < 22.5) / valid_pics)
        a5_error = 100.0 * (torch.sum(angle_error < 30) / valid_pics)

        # if valid_pics > 1e-5:
        # If the current node gets data with valid normal
        dist_node_cnt = (valid_pics - 1e-6) / valid_pics

        if is_distributed:
            dist.all_reduce(dist_node_cnt)
            dist.all_reduce(mean_error)
            dist.all_reduce(rmse_error)
            dist.all_reduce(a1_error)
            dist.all_reduce(a2_error)
            
            dist.all_reduce(a3_error)
            dist.all_reduce(a4_error)
            dist.all_reduce(a5_error)

        dist_node_cnt = dist_node_cnt.cpu().numpy()
        self.normal_mean.update(mean_error.cpu().numpy(), dist_node_cnt)
        self.normal_rmse.update(rmse_error.cpu().numpy(), dist_node_cnt)
        self.normal_a1.update(a1_error.cpu().numpy(), dist_node_cnt)
        self.normal_a2.update(a2_error.cpu().numpy(), dist_node_cnt)

        self.normal_median.update(median_error.cpu().numpy(), dist_node_cnt)
        self.normal_a3.update(a3_error.cpu().numpy(), dist_node_cnt)
        self.normal_a4.update(a4_error.cpu().numpy(), dist_node_cnt)
        self.normal_a5.update(a5_error.cpu().numpy(), dist_node_cnt)


    def get_metrics(self,):
        """
        """
        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict[metric] = self.__getattribute__(metric).avg
        return metrics_dict


    def get_metrics(self,):
        """
        """
        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict[metric] = self.__getattribute__(metric).avg
        return metrics_dict

def get_absrel_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes absolute relative error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    # Mean Absolute Relative Error
    rel = torch.abs(t_m - p_m) / (t_m + 1e-10) # compute errors
    abs_rel_sum = torch.sum(rel.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    abs_err = abs_rel_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)
    return torch.sum(abs_err), valid_pics

def get_sqrel_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes squared relative error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    # squared Relative Error
    sq_rel = torch.abs(t_m - p_m) ** 2 / (t_m + 1e-10) # compute errors
    sq_rel_sum = torch.sum(sq_rel.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    sqrel_err = sq_rel_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)
    return torch.sum(sqrel_err), valid_pics

def get_log10_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes log10 error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    diff_log = (torch.log10(p_m+1e-10) - torch.log10(t_m+1e-10)) * mask
    log10_diff = torch.abs(diff_log)
    log10_sum = torch.sum(log10_diff.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    log10_err = log10_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)
    return torch.sum(log10_err), valid_pics

def get_rmse_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes rmse error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    square = (t_m - p_m) ** 2
    rmse_sum = torch.sum(square.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    rmse = torch.sqrt(rmse_sum / (num + 1e-10))
    valid_pics = torch.sum(num > 0)
    return torch.sum(rmse), valid_pics

def get_rmse_log_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes log rmse error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    diff_log = (torch.log10(p_m+1e-10) - torch.log10(t_m+1e-10)) * mask
    square = diff_log ** 2
    rmse_log_sum = torch.sum(square.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    rmse_log = torch.sqrt(rmse_log_sum / (num + 1e-10))
    valid_pics = torch.sum(num > 0)
    return torch.sum(rmse_log), valid_pics

def get_silog_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes log rmse error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    diff_log = (torch.log10(p_m+1e-10) - torch.log10(t_m+1e-10)) * mask
    diff_log_sum = torch.sum(diff_log.reshape((b, c, -1)), dim=2) # [b, c]
    diff_log_square = diff_log ** 2
    diff_log_square_sum = torch.sum(diff_log_square.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    silog = torch.sqrt(diff_log_square_sum / (num + 1e-10) - (diff_log_sum / (num + 1e-10)) ** 2)
    valid_pics = torch.sum(num > 0)
    return torch.sum(silog), valid_pics

def get_ratio_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes the percentage of pixels for which the ratio of the two depth maps is less than a given threshold.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """
    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred

    gt_pred = t_m / (p_m + 1e-10)
    pred_gt = p_m / (t_m + 1e-10)
    gt_pred = gt_pred.reshape((b, c, -1))
    pred_gt = pred_gt.reshape((b, c, -1))
    gt_pred_gt = torch.cat((gt_pred, pred_gt), axis=1)
    ratio_max = torch.amax(gt_pred_gt, axis=1)

    delta_1_sum = torch.sum((ratio_max < 1.25), dim=1) # [b, ]
    delta_2_sum = torch.sum((ratio_max < 1.25 ** 2), dim=1) # [b, ]
    delta_3_sum = torch.sum((ratio_max < 1.25 ** 3), dim=1) # [b, ]
    num = torch.sum(mask.reshape((b, -1)), dim=1) # [b, ]

    delta_1 = delta_1_sum / (num + 1e-10)
    delta_2 = delta_2_sum / (num + 1e-10)
    delta_3 = delta_3_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)    

    return torch.sum(delta_1), torch.sum(delta_2), torch.sum(delta_3), valid_pics


if __name__ == '__main__':
    cfg = ['abs_rel', 'delta1']
    dam = MetricAverageMeter(cfg)

    pred_depth = np.random.random([2, 480, 640])
    gt_depth = np.random.random([2, 480, 640]) - 0.5
    intrinsic = [[100, 100, 200, 200], [200, 200, 300, 300]]
    
    pred = torch.from_numpy(pred_depth).cuda()
    gt = torch.from_numpy(gt_depth).cuda()

    mask = gt > 0
    dam.update_metrics_gpu(pred, gt, mask, False)
    eval_error = dam.get_metrics()
    print(eval_error)
    