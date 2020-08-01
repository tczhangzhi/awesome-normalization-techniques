import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


def convert_layers(model, layer_type_old, layer_type_new):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_layers(module, layer_type_old, layer_type_new)

        if type(module) == layer_type_old:
            # if pass other parameters change here such as:
            # layer_new = layer_type_new(num_groups, module.num_features, module.eps, module.affine)
            layer_new = layer_type_new(module.num_features)

            model._modules[name] = layer_new

    return model


if __name__ == '__main__':
    model = EfficientNet.from_pretrained('efficientnet-b2')
    convert_layers(model, nn.BatchNorm2d, nn.InstanceNorm2d)