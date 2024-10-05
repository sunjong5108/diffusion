import torch
import torch.nn as nn

def conv3x3(in_feat, out_ch):
    if len(in_feat.size()) == 3:
        ch, _, _ = in_feat.size()
    else:
        _, ch, _, _ = in_feat.size()

    conv_layer = nn.Conv2d(in_channels=ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1).to(in_feat.device)
    return conv_layer(in_feat)

def conv1x1(in_feat, out_ch):
    if len(in_feat.size()) == 3:
        ch, _, _ = in_feat.size()
    else:
        _, ch, _, _ = in_feat.size()

    conv_layer = nn.Conv2d(in_channels=ch, out_channels=out_ch, kernel_size=1, stride=1).to(in_feat.device)
    return conv_layer(in_feat)

def deconv(in_feat, out_ch):
    if len(in_feat.size()) == 3:
        ch, _, _ = in_feat.size()
    else:
        _, ch, _, _ = in_feat.size()

    deconv_layer = nn.ConvTranspose2d(in_channels=ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1).to(in_feat.device)
    return deconv_layer(in_feat)

def dense(in_feat, out_ch):
    if len(in_feat.size()) == 1:
        in_feat = in_feat.view(-1, 1)
        _, ch = in_feat.size()
    elif len(in_feat.size()) == 2:
        _, ch = in_feat.size()
    else:
        _, _, ch = in_feat.size()
    
    dense_layer = nn.Linear(in_features=ch, out_features=out_ch).to(in_feat.device)

    return dense_layer(in_feat)

def batchnorm(in_feat, out_ch):
    if len(in_feat.size()) == 2:
        bn = nn.BatchNorm1d(num_features=out_ch).to(in_feat.device)
    else:
        bn = nn.BatchNorm2d(num_features=out_ch).to(in_feat.device)
    return bn(in_feat)
