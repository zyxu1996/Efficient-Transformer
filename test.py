import os
import argparse
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np

from dataset import potsdam, label_to_RGB
from seg_metric import SegmentationMetric
import cv2
from mutil_scale_test import MultiEvalModule
import logging
import warnings


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


class params():
    def __init__(self, args2):
        if args2.dataset in ['potsdam', 'vaihingen']:
            self.number_of_classes = 6
        models = args2.models
        if models == 'HRNet_32':
            "hrnet32"
            self.STAGE2 = {'NUM_MODULES': 1,
                           'NUM_BRANCHES': 2,
                           'NUM_BLOCKS': [4, 4],
                           'NUM_CHANNELS': [32, 64],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}
            self.STAGE3 = {'NUM_MODULES': 4,
                           'NUM_BRANCHES': 3,
                           'NUM_BLOCKS': [4, 4, 4],
                           'NUM_CHANNELS': [32, 64, 128],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}
            self.STAGE4 = {'NUM_MODULES': 3,
                           'NUM_BRANCHES': 4,
                           'NUM_BLOCKS': [4, 4, 4, 4],
                           'NUM_CHANNELS': [32, 64, 128, 256],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}
        elif models == 'HRNet_48':
            self.STAGE2 = {'NUM_MODULES': 1,
                           'NUM_BRANCHES': 2,
                           'NUM_BLOCKS': [4, 4],
                           'NUM_CHANNELS': [32, 64],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}
            self.STAGE3 = {'NUM_MODULES': 4,
                           'NUM_BRANCHES': 3,
                           'NUM_BLOCKS': [4, 4, 4],
                           'NUM_CHANNELS': [32, 64, 128],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}
            self.STAGE4 = {'NUM_MODULES': 3,
                           'NUM_BRANCHES': 4,
                           'NUM_BLOCKS': [4, 4, 4, 4],
                           'NUM_CHANNELS': [32, 64, 128, 256],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--dataset", type=str, default='vaihingen', choices=['potsdam', 'vaihingen'])
    parser.add_argument("--val_batchsize", type=int, default=16)
    parser.add_argument("--crop_size", type=int, nargs='+', default=[512, 512], help='H, W')
    parser.add_argument("--models", type=str, default='danet',
                        choices=['danet', 'bisenetv2', 'pspnet', 'segbase', 'swinT', 'deeplabv3', 'fcn', 'fpn', 'unet', 'resT'])
    parser.add_argument("--head", type=str, default='uperhead')
    parser.add_argument("--use_edge", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default='work_dir')
    parser.add_argument("--base_dir", type=str, default='./')
    parser.add_argument("--information", type=str, default='RS')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--save_gpu_memory", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args2 = parser.parse_args()
    return args2


def get_model():
    models = args2.models
    if models == 'swinT':
        print(models, args2.head)
    else:
        print(models)
    if args2.dataset in ['potsdam', 'vaihingen']:
        nclass = 6
    assert models in ['danet', 'bisenetv2', 'pspnet', 'segbase', 'swinT', 'deeplabv3', 'fcn', 'fpn', 'unet', 'resT']
    if models == 'danet':
        from models.danet import DANet
        model = DANet(nclass=nclass, backbone='resnet50', pretrained_base=False)
    if models == 'bisenetv2':
        from models.bisenetv2 import BiSeNetV2
        model = BiSeNetV2(nclass=nclass)
    if models == 'pspnet':
        from models.pspnet import PSPNet
        model = PSPNet(nclass=nclass, backbone='resnet50', pretrained_base=False)
    if models == 'segbase':
        from models.segbase import SegBase
        model = SegBase(nclass=nclass, backbone='resnet50', pretrained_base=False)
    if models == 'swinT':
        from models.swinT import swin_tiny as swinT
        if args2.use_edge:
            model = swinT(nclass=nclass, pretrained=False, aux=True, head=args2.head, edge_aux=args2.use_edge)
        else:
            model = swinT(nclass=nclass, pretrained=False, aux=True, head=args2.head)
    if models == 'resT':
        from models.resT import rest_tiny as resT
        if args2.use_edge:
            model = resT(nclass=nclass, pretrained=False, aux=True, head=args2.head, edge_aux=args2.use_edge)
        else:
            model = resT(nclass=nclass, pretrained=False, aux=True, head=args2.head)
    if models == 'deeplabv3':
        from models.deeplabv3 import DeepLabV3
        model = DeepLabV3(nclass=nclass, backbone='resnet50', pretrained_base=False)
    if models == 'fcn':
        from models.fcn import FCN16s
        model = FCN16s(nclass=nclass)
    if models == 'fpn':
        from models.fpn import FPN
        model = FPN(nclass=nclass)
    if models == 'unet':
        from models.unet import UNet
        model = UNet(nclass=nclass)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args2.local_rank], output_device=args2.local_rank, find_unused_parameters=True)
    return model


args2 = parse_args()
args = params(args2)


cudnn.benchmark = True
cudnn.deterministic = False
cudnn.enabled = True
distributed = True

device = torch.device(('cuda:{}').format(args2.local_rank))


if distributed:
    torch.cuda.set_device(args2.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
    )

data_dir = os.path.join(args2.base_dir, 'data')
potsdam_val = potsdam(base_dir=data_dir, train=False,
                      dataset=args2.dataset, crop_szie=args2.crop_size)
if distributed:
    val_sampler = DistributedSampler(potsdam_val)
else:
    val_sampler = None
dataloader_val = DataLoader(
    potsdam_val,
    batch_size=args2.val_batchsize,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=val_sampler)

potsdam_val_full = potsdam(base_dir=data_dir, train=False,
                           dataset=args2.dataset, crop_szie=args2.crop_size, val_full_img=True)
if distributed:
    full_val_sampler = DistributedSampler(potsdam_val_full)
else:
    full_val_sampler = None
dataloader_val_full = DataLoader(
    potsdam_val_full,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=full_val_sampler)


def val(model, weight_path):

    if args2.dataset in ['potsdam', 'vaihingen']:
        nclasses = 6
    model.eval()
    metric = SegmentationMetric(numClass=nclasses)
    with torch.no_grad():
        model_state_file = weight_path
        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        else:
            warnings.warn('weight is not existed !!!"')

        for i, sample in enumerate(dataloader_val):

            images, labels = sample['image'], sample['label']
            images = images.cuda()
            labels = labels.long().squeeze(1)
            logits = model(images)
            print("test:{}/{}".format(i, len(dataloader_val)))
            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            metric.addBatch(logits, labels)
        result_count(metric)


def mutil_scale_val(model, weight_path, object_path):
    if args2.dataset in ['potsdam', 'vaihingen']:
        nclasses = 6
    model = MultiEvalModule(model, nclass=nclasses, flip=True, scales=[0.5, 0.75, 1.0, 1.25, 1.5], save_gpu_memory=args2.save_gpu_memory,
                            crop_size=args2.crop_size, stride_rate=1/2, get_batch=args2.val_batchsize)
    model.eval()
    metric = SegmentationMetric(nclasses)
    with torch.no_grad():
        model_state_file = weight_path
        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            elif 'model' in checkpoint:
                checkpoint = checkpoint['model']
            else:
                checkpoint = checkpoint
            checkpoint = {k: v for k, v in checkpoint.items() if not 'n_averaged' in k}
            checkpoint = {k.replace('model.', 'module.'): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        else:
            warnings.warn('weight is not existed !!!"')

        for i, sample in enumerate(dataloader_val_full):

            images, labels, names = sample['image'], sample['label'], sample['name']
            images = images.cuda()
            labels = labels.long().squeeze(1)
            logits = model(images)
            print("test:{}/{}".format(i, len(dataloader_val_full)))
            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            metric.addBatch(logits, labels)

            vis_logits = label_to_RGB(logits.squeeze())[:, :, ::-1]
            save_path = os.path.join(object_path, 'outputs', names[0] + '.png')
            cv2.imwrite(save_path, vis_logits)
        result_count(metric)


def result_count(metric):
    iou = metric.IntersectionOverUnion()
    miou = np.nanmean(iou[0:5])
    acc = metric.Accuracy()
    f1 = metric.F1()
    mf1 = np.nanmean(f1[0:5])
    precision = metric.Precision()
    mprecision = np.nanmean(precision[0:5])
    recall = metric.Recall()
    mrecall = np.nanmean(recall[0:5])

    iou = reduce_tensor(torch.from_numpy(np.array(iou)).to(device) / get_world_size()).cpu().numpy()
    miou = reduce_tensor(torch.from_numpy(np.array(miou)).to(device) / get_world_size()).cpu().numpy()
    acc = reduce_tensor(torch.from_numpy(np.array(acc)).to(device) / get_world_size()).cpu().numpy()
    f1 = reduce_tensor(torch.from_numpy(np.array(f1)).to(device) / get_world_size()).cpu().numpy()
    mf1 = reduce_tensor(torch.from_numpy(np.array(mf1)).to(device) / get_world_size()).cpu().numpy()
    precision = reduce_tensor(torch.from_numpy(np.array(precision)).to(device) / get_world_size()).cpu().numpy()
    mprecision = reduce_tensor(torch.from_numpy(np.array(mprecision)).to(device) / get_world_size()).cpu().numpy()
    recall = reduce_tensor(torch.from_numpy(np.array(recall)).to(device) / get_world_size()).cpu().numpy()
    mrecall = reduce_tensor(torch.from_numpy(np.array(mrecall)).to(device) / get_world_size()).cpu().numpy()

    if args2.local_rank == 0:
        print('\n')
        logging.info('####################### full image val ###########################')
        print('|{}:{}{}{}{}|'.format(str('CLASSES').ljust(24),
                                     str('Precision').rjust(10), str('Recall').rjust(10),
                                     str('F1').rjust(10), str('IOU').rjust(10)))
        logging.info('|{}:{}{}{}{}|'.format(str('CLASSES').ljust(24),
                                            str('Precision').rjust(10), str('Recall').rjust(10),
                                            str('F1').rjust(10), str('IOU').rjust(10)))
        for i in range(len(iou)):
            print('|{}:{}{}{}{}|'.format(str(CLASSES[i]).ljust(24),
                                         str(round(precision[i], 4)).rjust(10), str(round(recall[i], 4)).rjust(10),
                                         str(round(f1[i], 4)).rjust(10), str(round(iou[i], 4)).rjust(10)))
            logging.info('|{}:{}{}{}{}|'.format(str(CLASSES[i]).ljust(24),
                                                str(round(precision[i], 4)).rjust(10),
                                                str(round(recall[i], 4)).rjust(10),
                                                str(round(f1[i], 4)).rjust(10), str(round(iou[i], 4)).rjust(10)))
        print('mIoU:{} ACC:{} mF1:{} mPrecision:{} mRecall:{}'.format(round(miou * 100, 2),
                                                                      round(acc * 100, 2), round(mf1 * 100, 2),
                                                                      round(mprecision * 100, 2),
                                                                      round(mrecall * 100, 2)))
        logging.info('mIoU:{} ACC:{} mF1:{} mPrecision:{} mRecall:{}'.format(round(miou * 100, 2),
                                                                             round(acc * 100, 2), round(mf1 * 100, 2),
                                                                             round(mprecision * 100, 2),
                                                                             round(mrecall * 100, 2)))
        print('\n')


def get_model_path(args2):
    object_path, weight_path = None, None
    file_dir = os.path.join(args2.base_dir, args2.save_dir)
    file_list = os.listdir(file_dir)
    for file in file_list:
        if args2.models in file and args2.information in file:
            weight_path = os.path.join(file_dir, file, 'weights', 'best_weight.pkl')
            object_path = os.path.join(file_dir, file)
    if object_path is None or weight_path is None:
        tmp_path = os.path.join(file_dir, 'tmp_save')
        output_path = os.path.join(tmp_path, 'outputs')
        weight_path = os.path.join(tmp_path, 'weights')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        object_path = tmp_path
        weight_path = weight_path + '/best_weight.pkl'
        warnings.warn('path is not defined, will be set as "./work_dir/tmp_save"')
    return object_path, weight_path


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ.setdefault('RANK', '0')
    # os.environ.setdefault('WORLD_SIZE', '1')
    # os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    # os.environ.setdefault('MASTER_PORT', '29555')

    object_path, weight_path = get_model_path(args2)
    save_log = os.path.join(object_path, 'test.log')
    logging.basicConfig(filename=save_log, level=logging.INFO)

    if args2.dataset in ['potsdam', 'vaihingen']:
        CLASSES = ('Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter/background')

    model = get_model()
    # val(model, weight_path)
    mutil_scale_val(model, weight_path, object_path)






