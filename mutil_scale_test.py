###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import math
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel


up_kwargs = {'mode': 'bilinear', 'align_corners': False}


def module_inference(module, image, flip=True):
    if flip:
        h_img = h_flip_image(image)
        v_img = v_flip_image(image)
        img = torch.cat([image, h_img, v_img], dim=0)
        cat_output = module(img)
        if isinstance(cat_output, (list, tuple)):
            cat_output = cat_output[0]
        output, h_output, v_output = cat_output.chunk(3, dim=0)
        output = output + h_flip_image(h_output) + v_flip_image(v_output)
    else:
        output = module(image)
        if isinstance(output, (list, tuple)):
            output = output[0]

    return output


def resize_image(img, h, w, **up_kwargs):
    return F.upsample(img, (h, w), **up_kwargs)


def pad_image(img, crop_size):
    """crop_size could be list:[h, w] or int"""
    b,c,h,w = img.size()
    # assert(c==3)
    if len(crop_size) > 1:
        padh = crop_size[0] - h if h < crop_size[0] else 0
        padw = crop_size[1] - w if w < crop_size[1] else 0
    else:
        padh = crop_size - h if h < crop_size else 0
        padw = crop_size - w if w < crop_size else 0
    # pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    # for i in range(c):
        # note that pytorch pad params is in reversed orders
    min_padh = min(padh, h)
    min_padw = min(padw, w)
    if padw < w and padh < h:
        img_pad[:, :, :, :] = F.pad(img[:, :, :, :], (0, padw, 0, padh), mode='reflect')
    else:
        img_pad[:, :, 0:h + min_padh - 1, 0:w + min_padw - 1] = \
            F.pad(img[:, :, :, :], (0, min_padw - 1, 0, min_padh - 1), mode='reflect')

        img_pad[:, :, :, :] = F.pad(img_pad[:, :, 0:h + min_padh - 1, 0:w + min_padw - 1],
                                    (0, padw - min_padw + 1, 0, padh - min_padh + 1), mode='constant', value=0)
    if len(crop_size) > 1:
        assert (img_pad.size(2) >= crop_size[0] and img_pad.size(3) >= crop_size[1])
    else:
        assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad


def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]


def h_flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)


def v_flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(2, idx)


def hv_flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    img = img.index_select(3, idx)
    return img.index_select(2, idx)


class MultiEvalModule_Fullimg(DataParallel):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, device_ids=None, flip=True,
                 # scales=[1.0]):
                 # scales=[1.0,1.25]):
                 # scales=[0.5, 0.75,1.0,1.25,1.5]):
                 scales=[1.0]):
        super(MultiEvalModule_Fullimg, self).__init__(module, device_ids)
        self.nclass = nclass
        self.base_size = 256
        self.crop_size = 256
        self.scales = scales
        self.flip = flip
        print('MultiEvalModule_Fullimg: base_size {}, crop_size {}'. \
            format(self.base_size, self.crop_size))

    def forward(self, image):
        """Mult-size Evaluation"""
        batch, _, h, w = image.size()

        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch,self.nclass,h,w).zero_().cuda()
        for scale in self.scales:
            crop_size = int(math.ceil(self.crop_size * scale))

            cur_img = resize_image(image, crop_size, crop_size, **up_kwargs)
            outputs = module_inference(self.module, cur_img, self.flip)
            score = resize_image(outputs, h, w, **up_kwargs)
            scores += score

        return scores


class MultiEvalModule(nn.Module):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, device_ids=None, flip=True, save_gpu_memory=False,
                 scales=[1.0], get_batch=1, crop_size=[512, 512], stride_rate=1/2):
                 #scales=[0.5,0.75,1,1.25]):
                 #scales=[0.5,0.75,1.0,1.25,1.4,1.6,1.8]):
                 #scales=[1]):
        # super(MultiEvalModule, self).__init__(module, device_ids)
        super(MultiEvalModule, self).__init__()
        self.module = module
        self.devices_ids = device_ids
        self.nclass = nclass
        self.crop_size = np.array(crop_size)
        self.scales = scales
        self.flip = flip
        self.get_batch = get_batch
        self.stride_rate = stride_rate
        self.save_gpu_memory = save_gpu_memory  # if over memory, can try this

    def forward(self, image):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        batch, _, h, w = image.size()
        # assert(batch == 1)
        stride_rate = self.stride_rate
        with torch.cuda.device_of(image):
            if self.save_gpu_memory:
                scores = image.new().resize_(batch, self.nclass, h, w).zero_().cpu()
            else:
                scores = image.new().resize_(batch,self.nclass,h,w).zero_().cuda()

        for scale in self.scales:
            crop_size = self.crop_size
            stride = (crop_size * stride_rate).astype(np.int)

            if h > w:
                long_size = int(math.ceil(h * scale))
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                long_size = int(math.ceil(w * scale))
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height

            # resize image to current size
            cur_img = resize_image(image, height, width, **up_kwargs)
            if long_size <= np.max(crop_size):
                pad_img = pad_image(cur_img, crop_size)
                outputs = module_inference(self.module, pad_img, self.flip)
                outputs = crop_image(outputs, 0, height, 0, width)

            else:
                if short_size < np.min(crop_size):
                    # pad if needed
                    pad_img = pad_image(cur_img, crop_size)
                else:
                    pad_img = cur_img
                _,_,ph,pw = pad_img.size()
                # assert(ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph-crop_size[0])/stride[0])) + 1
                w_grids = int(math.ceil(1.0 * (pw-crop_size[1])/stride[1])) + 1
                with torch.cuda.device_of(image):
                    if self.save_gpu_memory:
                        outputs = image.new().resize_(batch, self.nclass, ph, pw).zero_().cpu()
                        count_norm = image.new().resize_(batch, 1, ph, pw).zero_().cpu()
                    else:
                        outputs = image.new().resize_(batch,self.nclass,ph,pw).zero_().cuda()
                        count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
                # grid evaluation
                location = []
                batch_size = []
                pad_img = pad_image(pad_img, [ph + crop_size[0], pw + crop_size[1]])  # expand pad_image

                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride[0]
                        w0 = idw * stride[1]
                        h1 = min(h0 + crop_size[0], ph)
                        w1 = min(w0 + crop_size[1], pw)

                        crop_img = crop_image(pad_img, h0, h0 + crop_size[0], w0, w0 + crop_size[1])
                        # pad if needed
                        pad_crop_img = pad_image(crop_img, crop_size)
                        size_h, size_w = pad_crop_img.shape[-2:]
                        pad_crop_img = resize_image(pad_crop_img, crop_size[0], crop_size[1], **up_kwargs)
                        if self.get_batch > 1:
                            location.append([h0, w0, h1, w1])
                            batch_size.append(pad_crop_img)
                            if len(location) == self.get_batch or (idh + idw + 2) == (h_grids + w_grids):
                                batch_size = torch.cat(batch_size, dim=0).cuda()
                                location = np.array(location)
                                output = module_inference(self.module, batch_size, self.flip)
                                output = output.detach()
                                output = resize_image(output, size_h, size_w, **up_kwargs)
                                if self.save_gpu_memory:
                                    output = output.detach().cpu()  # to save gpu memory
                                else:
                                    output = output.detach()
                                for i in range(batch_size.shape[0]):
                                    outputs[:, :, location[i][0]:location[i][2], location[i][1]:location[i][3]] += \
                                        crop_image(output[i, ...].unsqueeze(dim=0), 0, location[i][2]-location[i][0], 0, location[i][3]-location[i][1])
                                    count_norm[:, :, location[i][0]:location[i][2], location[i][1]:location[i][3]] += 1
                                location = []
                                batch_size = []
                        else:
                            output = module_inference(self.module, pad_crop_img, self.flip)
                            if self.save_gpu_memory:
                                output = output.detach().cpu()  # to save gpu memory
                            else:
                                output = output.detach()
                            output = resize_image(output, size_h, size_w, **up_kwargs)
                            outputs[:,:,h0:h1,w0:w1] += crop_image(output,
                                0, h1-h0, 0, w1-w0)
                            count_norm[:,:,h0:h1,w0:w1] += 1
                assert((count_norm==0).sum()==0)
                outputs = outputs / count_norm
                outputs = outputs[:,:,:height,:width]
            score = resize_image(outputs, h, w, **up_kwargs)
            scores += score
        return scores

