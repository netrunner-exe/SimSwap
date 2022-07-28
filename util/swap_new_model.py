# -*- coding: utf-8 -*-
# @Author: netrunner-exe
# @Date:   2022-07-01 13:45:41
# @Last Modified by:   netrunner-exe
# @Last Modified time: 2022-07-08 13:33:01
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from util.util import tensor2im


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def swap_result_new_model(face_align_crop, model, latend_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_align_crop = tensor2im(face_align_crop[0], imtype=np.uint8, normalize=False)
    img_align_crop = Image.fromarray(img_align_crop)

    img_tensor = transforms.ToTensor()(img_align_crop)
    img_tensor = img_tensor.view(-1, 3, img_align_crop.size[0], img_align_crop.size[1])

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
    
    # test = Image.fromarray(np.uint8(swap_result))
    # test.save("/content/simswap_img_result/face_JPG.jpg")
    # test.save("/content/simswap_img_result/face_PNG.png")   
    
    swap_result = _totensor(swap_result).to(device)

    return swap_result
