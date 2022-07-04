# -*- coding: utf-8 -*-
# @Author: netrunner-exe
# @Date:   2022-07-01 13:45:41
# @Last Modified by:   netrunner-exe
# @Last Modified time: 2022-07-04 14:25:16
import math

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import normalize
from torchvision.utils import make_grid

"""
img2tensor and tensor2img was taken from BasicSR
https://github.com/XPixelGroup/BasicSR

"""


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.
    After clamping to [min, max], values will be normalized to [0, 1].
    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.
    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def swap_result_new_model(face_align_crop, model, latend_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_align_crop = tensor2img(face_align_crop.squeeze(0), rgb2bgr=False, min_max=(-1, 1))
    img_align_crop = Image.fromarray(img_align_crop)

    img_tensor = transforms.ToTensor()(img_align_crop)
    img_tensor = img_tensor.view(-1, 3, img_align_crop.size[0], img_align_crop.size[1])
    img_tensor = normalize(img_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)

    img_tensor = img_tensor.to(device, non_blocking=True)
    img_tensor = img_tensor.sub_(mean).div_(std)

    imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

    swap_res = model.netG(img_tensor, latend_id).cpu()
    swap_res = (swap_res * imagenet_std + imagenet_mean).numpy()
    swap_res = swap_res.squeeze(0).transpose((1, 2, 0))

    swap_result = np.clip(255 * swap_res, 0, 255)
    swap_result = img2tensor(swap_result / 255., bgr2rgb=False, float32=True)
    return swap_result
