import torch.nn as nn
from layer import conv3x3, conv1x1, deconv, dense, batchnorm
from timeembedding import time_embedding

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embedding = time_embedding()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x, x_ts):
        x_ts = x_ts.float()
        x_ts = dense(x_ts, out_ch=192)
        x_ts = self.relu(x_ts)

        feat_1 = self.time_embedding(x, x_ts, 16)
        feat_1_down = self.maxpool(feat_1)
        
        feat_2 = self.time_embedding(feat_1_down, x_ts, 32)
        feat_2_down = self.maxpool(feat_2)
        
        feat_3 = self.time_embedding(feat_2_down, x_ts, 64)
        
        feat_3_flatten = self.flatten(feat_3)
        feat_3_concat = torch.cat((feat_3_flatten, x_ts), dim=1)
        feat_3_out = dense(feat_3_concat, 7*7*64)
        feat_3_out = batchnorm(feat_3_out, 7*7*64)
        feat_3_out = self.relu(feat_3_out)
        feat_3_out = feat_3_out.reshape(4, 64, 7, 7)
        
        feat_3 = torch.cat((feat_3_out, feat_3), dim=1)
        feat_3 = self.time_embedding(feat_3, x_ts, 64)
        feat_3_up = deconv(feat_3, 64)
        
        feat_2 = torch.cat((feat_3_up, feat_2), dim=1)
        feat_2 = self.time_embedding(feat_2, x_ts, 32)
        feat_2_up = deconv(feat_2, 32)
        
        feat_1 = torch.cat((feat_2_up, feat_1), dim=1)
        feat_1 = self.time_embedding(feat_1, x_ts, 16)
        
        feat_1 = conv1x1(feat_1, 1)
        
        return feat_1
