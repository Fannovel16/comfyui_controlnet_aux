import torch

from utils.flow_viz import flow_tensor_to_image
from .visualization import viz_depth_tensor


class Logger:
    def __init__(self, lr_scheduler,
                 summary_writer,
                 summary_freq=100,
                 start_step=0,
                 img_mean=None,
                 img_std=None,
                 ):
        self.lr_scheduler = lr_scheduler
        self.total_steps = start_step
        self.running_loss = {}
        self.summary_writer = summary_writer
        self.summary_freq = summary_freq

        self.img_mean = img_mean
        self.img_std = img_std

    def print_training_status(self, mode='train', is_depth=False):
        if is_depth:
            print('step: %06d \t loss: %.3f' % (self.total_steps, self.running_loss['total_loss'] / self.summary_freq))
        else:
            print('step: %06d \t epe: %.3f' % (self.total_steps, self.running_loss['epe'] / self.summary_freq))

        for k in self.running_loss:
            self.summary_writer.add_scalar(mode + '/' + k,
                                           self.running_loss[k] / self.summary_freq, self.total_steps)
            self.running_loss[k] = 0.0

    def lr_summary(self):
        lr = self.lr_scheduler.get_last_lr()[0]
        self.summary_writer.add_scalar('lr', lr, self.total_steps)

    def add_image_summary(self, img1, img2, flow_preds=None, flow_gt=None, mode='train',
                          is_depth=False,
                          ):
        if self.total_steps % self.summary_freq == 0:
            if is_depth:
                img1 = self.unnormalize_image(img1.detach().cpu())  # [3, H, W], range [0, 1]
                img2 = self.unnormalize_image(img2.detach().cpu())

                concat = torch.cat((img1, img2), dim=-1)  # [3, H, W*2]

                self.summary_writer.add_image(mode + '/img', concat, self.total_steps)
            else:
                img_concat = torch.cat((img1[0].detach().cpu(), img2[0].detach().cpu()), dim=-1)
                img_concat = img_concat.type(torch.uint8)  # convert to uint8 to visualize in tensorboard

                flow_pred = flow_tensor_to_image(flow_preds[-1][0])
                forward_flow_gt = flow_tensor_to_image(flow_gt[0])
                flow_concat = torch.cat((torch.from_numpy(flow_pred),
                                         torch.from_numpy(forward_flow_gt)), dim=-1)

                concat = torch.cat((img_concat, flow_concat), dim=-2)

                self.summary_writer.add_image(mode + '/img_pred_gt', concat, self.total_steps)

    def add_depth_summary(self, depth_pred, depth_gt, mode='train'):
        # assert depth_pred.dim() == 2  # [H, W]
        if self.total_steps % self.summary_freq == 0 or 'val' in mode:
            pred_viz = viz_depth_tensor(depth_pred.detach().cpu())  # [3, H, W]
            gt_viz = viz_depth_tensor(depth_gt.detach().cpu())

            concat = torch.cat((pred_viz, gt_viz), dim=-1)  # [3, H, W*2]

            self.summary_writer.add_image(mode + '/depth_pred_gt', concat, self.total_steps)

    def unnormalize_image(self, img):
        # img: [3, H, W], used for visualizing image
        mean = torch.tensor(self.img_mean).view(3, 1, 1).type_as(img)
        std = torch.tensor(self.img_std).view(3, 1, 1).type_as(img)

        out = img * std + mean

        return out

    def push(self, metrics, mode='train', is_depth=False, ):
        self.total_steps += 1

        self.lr_summary()

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.summary_freq == 0:
            self.print_training_status(mode, is_depth=is_depth)
            self.running_loss = {}

    def write_dict(self, results):
        for key in results:
            tag = key.split('_')[0]
            tag = tag + '/' + key
            self.summary_writer.add_scalar(tag, results[key], self.total_steps)

    def close(self):
        self.summary_writer.close()
