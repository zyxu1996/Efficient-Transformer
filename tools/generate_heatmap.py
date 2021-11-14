import cv2
import os
import numpy as np
from models.swinT import swin_base
import torch
import tifffile as tiff
from tools.utils import label_to_RGB
from torchvision import transforms


def init_model():
    model = swin_base(nclass=6, aux=True, head='uperhead')
    weight_dir = '../work_dir/' \
                 'swinT_lr0.0003_epoch100_batchsize16_swinT_upernet_base_imagenetpretrain_aux_AdamW_num73' \
                 '/weights/best_weight.pkl'
    checkpoint = torch.load(weight_dir, map_location=lambda storage, loc: storage)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = {k.replace('module.model.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    return model


def read_img(save_dir):
    img_dir = '../data/vaihingen/images/top_mosaic_09cm_area10.tif'
    image = tiff.imread(img_dir)
    image = image[1000:1000 + 512, 1000:1000+512, :]
    cv2.imwrite(os.path.join(save_dir, 'ori.png'), image[..., ::-1])

    return image


def to_tensor(image):
    image = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
    normalize = transforms.Normalize((.485, .456, .406), (.229, .224, .225))
    image = normalize(image).unsqueeze(0)

    return image


def main():
    save_img_dir = os.path.join(save_path, 'origin_img')
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    save_out_dir = os.path.join(save_path, 'output')
    if not os.path.exists(save_out_dir):
        os.mkdir(save_out_dir)

    image = read_img(save_img_dir)
    image = to_tensor(image).cuda()
    model = init_model().cuda().eval()
    with torch.no_grad():
        output = model(image)
    output = torch.argmax(output[0], dim=1)
    output = output.squeeze()
    output = output.cpu().numpy()
    output = output.astype(np.uint8)
    output = label_to_RGB(output)
    cv2.imwrite(os.path.join(save_out_dir, 'out.png'), output[..., ::-1])


if __name__ == '__main__':
    save_path = './heatmap/uperhead/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    main()

