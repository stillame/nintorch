#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torchvision import models


__all__ = [
    'load_pretrain_from_torch',
    'replace_layer_from_name',
    'filter_state_dict',]


def load_pretrain_from_torch(
        net_name: str, pretrained: bool = True):
    """Load the pretain model from pytorch pretrained models.
    pretrained: if True downloading pretrained weight directly,
        Otherwise using the model defintation.
    """
    net_name = net_name.lower()
    # TODO: adding supported_net to support more than vgg16.
    # The name of the supported_net should be the same as the method in pytorch.
    supported_net = ['vgg16', 'vgg16_bn']

    if net_name in supported_net:
        pretain_model = getattr(models, net_name)(pretrained=pretrained)
    else:
        raise NotImplementedError(f'Not in support list: {supported_net}')
    return pretain_model


def replace_layer_from_name(
        model, name_to_replace: str, layer_to_replace_with):
    """
    """
    model_with_replace = setattr(model, name_to_replace, layer_to_replace_with)
    return model_with_replace


def filter_state_dict(model, filter_kw: str):
    """Using filter_kw to filter out the state_dict.
    After using this then please using method load_state_dict from Module to load the pretrain model.
    """
    name_without_kw = list(
        filter(lambda x: x.find(filter_kw) < 0, model.state_dict()))
    filtered_state_dict = {i: model.state_dict()[i] for i in name_without_kw}
    return filtered_state_dict
