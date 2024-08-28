#https://github.com/SkyTNT/anime-segmentation/tree/main
#Only adapt isnet_is (https://huggingface.co/skytnt/anime-seg/blob/main/isnetis.ckpt)
import torch.nn as nn
import torch
from .isnet import ISNetDIS
import numpy as np
import cv2
from comfy.model_management import get_torch_device
DEVICE = get_torch_device()

class AnimeSegmentation:
    def __init__(self, ckpt_path):
        super(AnimeSegmentation).__init__()
        sd = torch.load(ckpt_path, map_location="cpu")
        self.net = ISNetDIS()
        #gt_encoder isn't used during inference
        self.net.load_state_dict({k.replace("net.", ''):v for k, v in sd.items() if k.startswith("net.")})
        self.net = self.net.to(DEVICE)
        self.net.eval()
    
    def get_mask(self, input_img, s=640):
        input_img = (input_img / 255).astype(np.float32)
        if s == 0:
            img_input = np.transpose(input_img, (2, 0, 1))
            img_input = img_input[np.newaxis, :]
            tmpImg = torch.from_numpy(img_input).float().to(DEVICE)
            with torch.no_grad():
                pred = self.net(tmpImg)[0][0].sigmoid() #https://github.com/SkyTNT/anime-segmentation/blob/main/train.py#L92C20-L92C47
                pred = pred.cpu().numpy()[0]
                pred = np.transpose(pred, (1, 2, 0))
                #pred = pred[:, :, np.newaxis]
                return pred

        h, w = h0, w0 = input_img.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        tmpImg = torch.from_numpy(img_input).float().to(DEVICE)
        with torch.no_grad():
            pred = self.net(tmpImg)[0][0].sigmoid() #https://github.com/SkyTNT/anime-segmentation/blob/main/train.py#L92C20-L92C47
            pred = pred.cpu().numpy()[0]
            pred = np.transpose(pred, (1, 2, 0))
            pred = pred[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
            #pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
            pred = cv2.resize(pred, (w0, h0))
            return pred

    def __call__(self, np_img, img_size):
        mask = self.get_mask(np_img, int(img_size))
        np_img = (mask * np_img + 255 * (1 - mask)).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)
        #np_img = np.concatenate([np_img, mask], axis=2, dtype=np.uint8)
        #mask = mask.repeat(3, axis=2)
        return mask, np_img

