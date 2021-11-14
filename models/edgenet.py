import os
import torch
from models.resT import rest_tiny


def EdgeNet(nclass=6):
    model = rest_tiny(nclass=nclass, pretrained=True, aux=True, edge_aux=False, head='mlphead')
    return model


def edgenet_init(weight_dir):
    with torch.no_grad():
        model = rest_tiny(nclass=6, pretrained=False, aux=True, edge_aux=False, head='mlphead').eval()
        if os.path.isfile(weight_dir):
            print('loaded edge model successfully')
            checkpoint = torch.load(weight_dir, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('module.model.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
    return model


if __name__ == '__main__':
    from tools.flops_params_fps_count import flops_params_fps
    model = EdgeNet(nclass=6)
    flops_params_fps(model)











