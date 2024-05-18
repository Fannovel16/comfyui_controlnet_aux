import importlib 
import torch
import torch.distributed as dist
from .avg_meter import AverageMeter
from collections import defaultdict, OrderedDict
import os
import socket
from custom_mmpkg.custom_mmcv.utils import collect_env as collect_base_env
try:
    from custom_mmpkg.custom_mmcv.utils import get_git_hash
except:
    from mmengine.utils import get_git_hash
#import mono.mmseg as mmseg
# import mmseg
import time
import datetime
import logging


def main_process() -> bool:
    return get_rank() == 0
    #return not cfg.distributed or \
    #       (cfg.distributed and cfg.local_rank == 0)

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def _find_free_port():
    # refer to https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port 

def _is_free_port(port):
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


# def collect_env():
#     """Collect the information of the running environments."""
#     env_info = collect_base_env()
#     env_info['MMSegmentation'] = f'{mmseg.__version__}+{get_git_hash()[:7]}'

#     return env_info

def init_env(launcher, cfg):
    """Initialize distributed training environment.
    If argument ``cfg.dist_params.dist_url`` is specified as 'env://', then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    """
    if launcher == 'slurm':
        _init_dist_slurm(cfg)
    elif launcher == 'ror':
        _init_dist_ror(cfg)
    elif launcher == 'None':
        _init_none_dist(cfg)
    else:
        raise RuntimeError(f'{cfg.launcher} has not been supported!')

def _init_none_dist(cfg):
    cfg.dist_params.num_gpus_per_node = 1
    cfg.dist_params.world_size = 1
    cfg.dist_params.nnodes = 1
    cfg.dist_params.node_rank = 0
    cfg.dist_params.global_rank = 0
    cfg.dist_params.local_rank = 0
    os.environ["WORLD_SIZE"] = str(1)

def _init_dist_ror(cfg):
    from ac2.ror.comm import get_local_rank, get_world_rank, get_local_size, get_node_rank, get_world_size
    cfg.dist_params.num_gpus_per_node = get_local_size()
    cfg.dist_params.world_size = get_world_size()
    cfg.dist_params.nnodes = (get_world_size()) // (get_local_size())
    cfg.dist_params.node_rank = get_node_rank()
    cfg.dist_params.global_rank = get_world_rank()
    cfg.dist_params.local_rank = get_local_rank()
    os.environ["WORLD_SIZE"] = str(get_world_size())


def _init_dist_slurm(cfg):
    if 'NNODES' not in os.environ:
        os.environ['NNODES'] = str(cfg.dist_params.nnodes)
    if 'NODE_RANK' not in os.environ:
        os.environ['NODE_RANK'] = str(cfg.dist_params.node_rank)

    #cfg.dist_params.
    num_gpus = torch.cuda.device_count()
    world_size = int(os.environ['NNODES']) * num_gpus
    os.environ['WORLD_SIZE'] = str(world_size)

    # config port
    if 'MASTER_PORT' in os.environ:
        master_port = str(os.environ['MASTER_PORT'])  # use MASTER_PORT in the environment variable
    else:
        # if torch.distributed default port(29500) is available
        # then use it, else find a free port
        if _is_free_port(16500):
            master_port = '16500'
        else:
            master_port = str(_find_free_port())
        os.environ['MASTER_PORT'] = master_port

    # config addr
    if 'MASTER_ADDR' in os.environ:
        master_addr = str(os.environ['MASTER_PORT'])  # use MASTER_PORT in the environment variable
    # elif cfg.dist_params.dist_url is not None:
    #     master_addr = ':'.join(str(cfg.dist_params.dist_url).split(':')[:2])
    else:
        master_addr = '127.0.0.1' #'tcp://127.0.0.1'
        os.environ['MASTER_ADDR'] = master_addr

    # set dist_url to 'env://' 
    cfg.dist_params.dist_url =  'env://' #f"{master_addr}:{master_port}"
    
    cfg.dist_params.num_gpus_per_node = num_gpus
    cfg.dist_params.world_size = world_size
    cfg.dist_params.nnodes = int(os.environ['NNODES'])
    cfg.dist_params.node_rank = int(os.environ['NODE_RANK'])
        
    # if int(os.environ['NNODES']) > 1 and cfg.dist_params.dist_url.startswith("file://"):
    #     raise Warning("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")
        

def get_func(func_name):
    """
        Helper to return a function object by name. func_name must identify 
        a function in this module or the path to a function relative to the base
        module.
        @ func_name: function name.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except:
        raise RuntimeError(f'Failed to find function: {func_name}')

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

class TrainingStats(object):
    """Track vital training statistics."""
    def __init__(self, log_period, tensorboard_logger=None):
        self.log_period = log_period
        self.tblogger = tensorboard_logger
        self.tb_ignored_keys = ['iter', 'eta', 'epoch', 'time']
        self.iter_timer = Timer()
        # Window size for smoothing tracked values (with median filtering)
        self.filter_size = log_period
        def create_smoothed_value():
            return AverageMeter()
        self.smoothed_losses = defaultdict(create_smoothed_value)
        #self.smoothed_metrics = defaultdict(create_smoothed_value)
        #self.smoothed_total_loss = AverageMeter()


    def IterTic(self):
        self.iter_timer.tic()

    def IterToc(self):
        return self.iter_timer.toc(average=False)

    def reset_iter_time(self):
        self.iter_timer.reset()

    def update_iter_stats(self, losses_dict):
        """Update tracked iteration statistics."""
        for k, v in losses_dict.items():
            self.smoothed_losses[k].update(float(v), 1)

    def log_iter_stats(self, cur_iter, optimizer, max_iters, val_err={}):
        """Log the tracked statistics."""
        if (cur_iter % self.log_period == 0):
            stats = self.get_stats(cur_iter, optimizer, max_iters, val_err)
            log_stats(stats)
            if self.tblogger:
                self.tb_log_stats(stats, cur_iter)
            for k, v in self.smoothed_losses.items():
                v.reset()

    def tb_log_stats(self, stats, cur_iter):
        """Log the tracked statistics to tensorboard"""
        for k in stats:
            # ignore some logs
            if k not in self.tb_ignored_keys:
                v = stats[k]
                if isinstance(v, dict):
                    self.tb_log_stats(v, cur_iter)
                else:
                    self.tblogger.add_scalar(k, v, cur_iter)


    def get_stats(self, cur_iter, optimizer, max_iters, val_err = {}):
        eta_seconds = self.iter_timer.average_time * (max_iters - cur_iter)

        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        stats = OrderedDict(
            iter=cur_iter,  # 1-indexed
            time=self.iter_timer.average_time,
            eta=eta,
        )
        optimizer_state_dict = optimizer.state_dict()
        lr = {}
        for i in range(len(optimizer_state_dict['param_groups'])):
            lr_name = 'group%d_lr' % i
            lr[lr_name] = optimizer_state_dict['param_groups'][i]['lr']

        stats['lr'] = OrderedDict(lr)
        for k, v in self.smoothed_losses.items():
            stats[k] = v.avg

        stats['val_err'] = OrderedDict(val_err)
        stats['max_iters'] = max_iters
        return stats


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.
    Args:
        @input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        @average (bool): whether to do average or sum
    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def log_stats(stats):
    logger = logging.getLogger()
    """Log training statistics to terminal"""
    lines = "[Step %d/%d]\n" % (
            stats['iter'], stats['max_iters'])

    lines += "\t\tloss: %.3f,    time: %.6f,    eta: %s\n" % (
        stats['total_loss'], stats['time'], stats['eta'])

    # log loss
    lines += "\t\t" 
    for k, v in stats.items():
        if 'loss' in k.lower() and 'total_loss' not in k.lower():
            lines += "%s: %.3f" % (k, v)  + ",  "
    lines = lines[:-3]
    lines += '\n'

    # validate criteria
    lines += "\t\tlast val err:" + ",  ".join("%s: %.6f" % (k, v) for k, v in stats['val_err'].items()) + ", "
    lines += '\n'

    # lr in different groups
    lines += "\t\t" +  ",  ".join("%s: %.8f" % (k, v) for k, v in stats['lr'].items())
    lines += '\n'
    logger.info(lines[:-1])  # remove last new linen_pxl

