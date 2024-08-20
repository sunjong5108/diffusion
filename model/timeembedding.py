import torch.nn as nn
from layer import conv3x3, dense

class time_embedding(nn.Module):
    def __init__(self, out_ch):
        super().__init__()

        self.out_ch = out_ch
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(num_features=out_ch)

    def forward(self, x_img, x_ts):
        out_ch = self.out_ch
        x_parameter = conv3x3(x_img, out_ch)
        x_parameter = self.relu(x_parameter)

        x_ts = x_ts.view(-1, 1, 1).float()
        time_parameter = dense(x_ts, out_ch)
        time_parameter = self.relu(time_parameter)
        time_parameter = time_parameter.view(-1, out_ch, 1, 1)
        x_parameter = x_parameter * time_parameter

        x_out = conv3x3(x_parameter, out_ch)
        x_out = x_out + x_parameter
        x_out = self.batchnorm(x_out)
        x_out = self.relu(x_out)

        return x_out