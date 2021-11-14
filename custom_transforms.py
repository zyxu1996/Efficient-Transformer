import torch
import random
import numpy as np
import cv2
import os
import torch.nn as nn
from torchvision import transforms

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)

        return {'image': image, 'label': label}


class RandomScaleCrop(object):
    def __init__(self, base_size=None, crop_size=None, fill=0):
        """shape [H, W]"""
        if base_size is None:
            base_size = [512, 512]
        if crop_size is None:
            crop_size = [512, 512]
        self.base_size = np.array(base_size)
        self.crop_size = np.array(crop_size)
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.choice([self.base_size * 0.5, self.base_size * 0.75, self.base_size,
                                    self.base_size * 1.25, self.base_size * 1.5])
        short_size = short_size.astype(np.int)
        h, w = img.shape[0:2]
        if h > w:
            ow = short_size[1]
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size[0]
            ow = int(1.0 * w * oh / h)
        #img = img.resize((ow, oh), Image.BILINEAR)
        #mask = mask.resize((ow, oh), Image.NEAREST)
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        # pad crop
        if short_size[0] < self.crop_size[0] or short_size[1] < self.crop_size[1]:
            padh = self.crop_size[0] - oh if oh < self.crop_size[0] else 0
            padw = self.crop_size[1] - ow if ow < self.crop_size[1] else 0
            #img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            #mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
            img = cv2.copyMakeBorder(img, 0, padh, 0, padw, borderType=cv2.BORDER_DEFAULT)
            mask = cv2.copyMakeBorder(mask, 0, padh, 0, padw, borderType=cv2.BORDER_DEFAULT)
        # random crop crop_size
        h, w = img.shape[0:2]
        x1 = random.randint(0, w - self.crop_size[1])
        y1 = random.randint(0, h - self.crop_size[0])
        img = img[y1:y1+self.crop_size[0], x1:x1+self.crop_size[1], :]
        mask = mask[y1:y1+self.crop_size[0], x1:x1+self.crop_size[1]]
        return {'image': img, 'label': mask}


class ImageSplit(nn.Module):
    def __init__(self, numbers=None):
        super(ImageSplit, self).__init__()
        """numbers [H, W]
        split from left to right, top to bottom"""
        if numbers is None:
            numbers = [2, 2]
        self.num = numbers

    def forward(self, x):
        flag = None
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=1)
            flag = 1
        b, c, h, w = x.shape
        num_h, num_w = self.num[0], self.num[1]
        assert h % num_h == 0 and w % num_w == 0
        split_h, split_w = h // num_h, w // num_w

        outputs = []
        outputss = []
        for i in range(b):
            for h_i in range(num_h):
                for w_i in range(num_w):
                    output = x[i][:, split_h * h_i: split_h * (h_i + 1),
                             split_w * w_i: split_w * (w_i + 1)].unsqueeze(dim=0)
                    outputs.append(output)
            outputs = torch.cat(outputs, dim=0).unsqueeze(dim=0)
            outputss.append(outputs)
            outputs = []
        outputss = torch.cat(outputss, dim=0).contiguous()
        if flag is not None:
            outputss = outputss.squeeze(dim=2)
        return outputss


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, add_edge=True):
        """imagenet normalize"""
        self.normalize = transforms.Normalize((.485, .456, .406), (.229, .224, .225))
        self.add_edge = add_edge

    def get_edge(self, img, edge_width=3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        edge = cv2.Canny(gray, 50, 150)
        # cv2.imshow('edge', edge)
        # cv2.waitKey(0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
        edge = cv2.dilate(edge, kernel)
        edge = edge / 255
        edge = torch.from_numpy(edge).unsqueeze(dim=0).float()

        return edge

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']

        mask = np.expand_dims(mask, axis=2)
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.int64).transpose((2, 0, 1))

        img = torch.from_numpy(img).float().div(255)
        img = self.normalize(img)
        mask = torch.from_numpy(mask).float()

        if self.add_edge:
            edge = self.get_edge(sample['image'])
            img = img + edge

        return {'image': img, 'label': mask}


class RGBGrayExchange():
    def __init__(self, path=None, palette=None):
        self.palette = palette
        """RGB format"""
        if palette is None:
            self.palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255],
                       [0, 255, 0], [255, 255, 0], [255, 0, 0]]
        self.path = path

    def read_img(self):
        img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            img = img[:, :, ::-1]
        return img

    def RGB_to_Gray(self, image=None):
        if not self.path is None:
            image = self.read_img()
        Gray = np.zeros(shape=[image.shape[0], image.shape[1]], dtype=np.uint8)
        for i in range(len(self.palette)):
            index = image == np.array(self.palette[i])
            index[..., 0][index[..., 1] == False] = False
            index[..., 0][index[..., 2] == False] = False
            Gray[index[..., 0]] = i
        print('unique pixels:{}'.format(np.unique(Gray)))
        return Gray

    def Gray_to_RGB(self, image=None):
        if not self.path is None:
            image = self.read_img()
        RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
        for i in range(len(self.palette)):
            index = image == i
            RGB[index] = np.array(self.palette[i])
        print('unique pixels:{}'.format(np.unique(RGB)))
        return RGB


class Mixup(nn.Module):
    def __init__(self, alpha=1.0, use_edge=False):
        super(Mixup, self).__init__()
        self.alpha = alpha
        self.use_edge = use_edge

    def criterion(self, lam, outputs, targets_a, targets_b, criterion):
        return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

    def forward(self, inputs, targets, criterion, model):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size).cuda()
        mix_inputs = lam*inputs + (1-lam)*inputs[index, :]
        targets_a, targets_b = targets, targets[index]
        outputs = model(mix_inputs)

        losses = 0
        if isinstance(outputs, (list, tuple)):
                if self.use_edge:
                    for i in range(len(outputs) - 1):
                        loss = self.criterion(lam, outputs[i], targets_a, targets_b, criterion[0])
                        losses += loss
                    edge_targets_a = edge_contour(targets).long()
                    edge_targets_b = edge_targets_a[index]
                    loss2 = self.criterion(lam, outputs[-1], edge_targets_a, edge_targets_b, criterion[1])
                    losses += loss2
                else:
                    for i in range(len(outputs)):
                        loss = self.criterion(lam, outputs[i], targets_a, targets_b, criterion)
                        losses += loss
        else:
            losses = self.criterion(lam, outputs, targets_a, targets_b, criterion)
        return losses


def edge_contour(label, edge_width=3):
    import cv2
    cuda_type = label.is_cuda
    label = label.cpu().numpy().astype(np.int)
    b, h, w = label.shape
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[:, 1:h, :]
    edge_right[(label[:, 1:h, :] != label[:, :h - 1, :]) & (label[:, 1:h, :] != 255)
               & (label[:, :h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :, :w - 1]
    edge_up[(label[:, :, :w - 1] != label[:, :, 1:w])
            & (label[:, :, :w - 1] != 255)
            & (label[:, :, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:, :h - 1, :w - 1]
    edge_upright[(label[:, :h - 1, :w - 1] != label[:, 1:h, 1:w])
                 & (label[:, :h - 1, :w - 1] != 255)
                 & (label[:, 1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:, :h - 1, 1:w]
    edge_bottomright[(label[:, :h - 1, 1:w] != label[:, 1:h, :w - 1])
                     & (label[:, :h - 1, 1:w] != 255)
                     & (label[:, 1:h, :w - 1] != 255)] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    for i in range(edge.shape[0]):
        edge[i] = cv2.dilate(edge[i], kernel)

    # edge[edge == 1] = 255     # view edge
    # import random
    # cv2.imwrite(os.path.join('./edge',  '{}.png'.format(random.random())), edge[0])
    if cuda_type:
        edge = torch.from_numpy(edge).cuda()
    else:
        edge = torch.from_numpy(edge)

    return edge


if __name__ == '__main__':
    path = './data/vaihingen/annotations/labels'
    filelist = os.listdir(path)
    for file in filelist:
        print(file)
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_UNCHANGED)
        img = torch.from_numpy(img).unsqueeze(dim=0).repeat(2, 1, 1)
        img = edge_contour(img)
        # cv2.imwrite(os.path.join(save_path, os.path.splitext(file)[0] + '.png'), gray)
