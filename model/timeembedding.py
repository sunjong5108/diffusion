import torch.nn as nn
from layer import conv3x3, dense, batchnorm

class time_embedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

    def forward(self, x_img, x_ts, out_ch):
        x_parameter = conv3x3(x_img, out_ch)
        x_parameter = self.relu(x_parameter)
        x_ts = x_ts.float()
        time_parameter = dense(x_ts, out_ch)
        time_parameter = self.relu(time_parameter)
        time_parameter = time_parameter.view(-1, out_ch, 1, 1)
        x_parameter = x_parameter * time_parameter

        x_out = conv3x3(x_parameter, out_ch)
        x_out = x_out + x_parameter
        x_out = batchnorm(x_out, out_ch)
        x_out = self.relu(x_out)

        return x_out
