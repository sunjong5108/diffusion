import torch
import torch.nn as nn

def conv3x3(in_feat, out_ch):
    if len(in_feat.size()) == 3:
        ch, _, _ = in_feat.size()
    else:
        _, ch, _, _ = in_feat.size()

    conv_layer = nn.Conv2d(in_channels=ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1).to(in_feat.device)
    return conv_layer(in_feat)

def dense(in_feat, out_ch):
    if len(in_feat.size()) == 2:
        _, ch = in_feat.size()
    else:
        _, _, ch = in_feat.size()
    
    dense_layer = nn.Linear(in_features=ch, out_features=out_ch).to(in_feat.device)

    return dense_layer(in_feat)