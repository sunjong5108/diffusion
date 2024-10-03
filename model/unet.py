import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.time_embedding = time_embedding()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x, x_ts):
        x_ts = dense(x_ts, out_ch=192)
        x_ts = self.relu(x_ts)

        feat_1 = self.time_embedding(x, x_ts, 16)
        feat_1 = self.maxpool(feat_1)
        
        feat_2 = self.time_embedding(feat_1, x_ts, 32)
        feat_2 = self.maxpool(feat_2)
        
        feat_3 = self.time_embedding(feat_2, x_ts, 64)
        
        feat_3_flatten = self.flatten(feat3)
        feat_3_concat = torch.cat((feat_3_flatten, x_ts))
        feat_3_out = dense(feat_3_concat, 7*7*64)
        feat_3_out = batchnorm(feat_3_out, 7*7*64)
        feat_3_out = self.relu(feat_3_out)
        feat_3_out = feat_3_out.reshape(64, 7, 7)
        
        feat_3 = torch.cat((feat_3_out, feat_3))
        feat_3 = time_embedding(feat_3, x_ts, 64)
        feat_3 = time_embedding(feat_3, x_ts, 64)
        
        
        '''
        작성 중
        '''
        
        
        

        return
