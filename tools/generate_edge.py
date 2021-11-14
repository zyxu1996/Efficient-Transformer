import cv2
import tifffile as tiff
import os
import numpy as np
from tools.utils import label_to_RGB
from models.resT import rest_tiny
import torch
from torchvision import transforms


def to_tensor(image):
    image = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
    normalize = transforms.Normalize((.485, .456, .406), (.229, .224, .225))
    image = normalize(image).unsqueeze(0)

    return image


def init_model():
    model = rest_tiny(nclass=6, aux=False, head='mlphead', edge_aux=True)
    weight_dir = '../work_dir/' \
                 'resT_lr0.0003_epoch100_batchsize16_vaihingen_resT_tiny_mlphead_imagenetpretrain_noaux_edge_edgeup_AdamW_num128' \
                 '/weights/best_weight.pkl'
    checkpoint = torch.load(weight_dir, map_location=lambda storage, loc: storage)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = {k.replace('module.model.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    return model


def read_img_label(save_dir):
    img_dir = '../data/vaihingen/images/top_mosaic_09cm_area10.tif'
    label_dir = '../data/vaihingen/annotations/labels/top_mosaic_09cm_area10.png'
    image = tiff.imread(img_dir)
    image = image[1000:1000 + 512, 1000:1000+512, :]
    label = cv2.imread(label_dir, cv2.IMREAD_UNCHANGED)
    label = label[1000:1000 + 512, 1000:1000 + 512]
    cv2.imwrite(os.path.join(save_dir, 'ori_img.png'), image[..., ::-1])
    cv2.imwrite(os.path.join(save_dir, 'ori_label.png'), label_to_RGB(label)[..., ::-1])

    return image, label


def canny_edge(img, edge_width=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    edge = cv2.Canny(gray, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)

    return edge


def groundtruth_edge(label, edge_width=3):
    if len(label.shape) == 2:
        label = label[np.newaxis, ...]
    label = label.astype(np.int)
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
    edge = edge.squeeze(axis=0)
    return edge


def get_edge_predict(img):
    img = to_tensor(img).cuda()
    model = init_model().cuda().eval()
    with torch.no_grad():
        output = model(img)
    edge_predict = torch.argmax(output[1], dim=1)
    edge_predict = edge_predict.squeeze().cpu().numpy().astype(np.uint8)
    edge_predict = edge_predict * 255
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # edge_predict = cv2.erode(edge_predict, kernel)

    return edge_predict


def main():
    img, label = read_img_label(save_path)
    # canny_ = canny_edge(img)
    # cv2.imwrite(os.path.join(save_path, 'canny_edge.png'), canny_)
    # groundtruth_ = groundtruth_edge(label) * 255
    # cv2.imwrite(os.path.join(save_path, 'groundtruth_edge.png'), groundtruth_)
    edge_predict = get_edge_predict(img)
    cv2.imwrite(os.path.join(save_path, 'predict_edge.png'), edge_predict)


if __name__ == '__main__':
    save_path = './edge/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    main()

