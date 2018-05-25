import torch
from torch import nn

def conv_relu_bn(n_in, n_out, kernel_size=3, stride=1, padding=0, dilation=1):
    """
    n_time_out = floor((n_time_in + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    """
    return nn.Sequential(
        nn.Conv1d(n_in, n_out, kernel_size, stride, padding, dilation),
        nn.BatchNorm1d(n_out),
        nn.ReLU()
    )


class SimpleTDNN(nn.Module):
    def __init__(self, n_pdf, n_freq=40, n_aux=100, n_time=29, n_stride=3, n_unit=512):
        """
        total kernel width should be 29 and stride 3
        """
        super().__init__()
        # T/3
        self.input_layer = conv_relu_bn(n_freq, n_unit, 5, stride=3)
        self.aux_layer = nn.Sequential(
            nn.Linear(n_aux, n_unit),
            nn.ReLU()
        )
        self.common = nn.Sequential(
            conv_relu_bn(n_unit, n_unit, 1),
            # 3*2=6
            conv_relu_bn(n_unit, n_unit, 3),
            conv_relu_bn(n_unit, n_unit, 1),
            # 3*(2+2)=12
            conv_relu_bn(n_unit, n_unit, 3),
            conv_relu_bn(n_unit, n_unit, 1),
            # 3*(2+2+3*2)=30
            conv_relu_bn(n_unit, n_unit, 3, 1, 0, 2),
            # conv_relu_bn(n_unit, n_unit, 3, 1, 0, 2),
            # conv_relu_bn(n_unit, n_unit, 3, 1, 0, 2),
            conv_relu_bn(n_unit, n_unit, 1),
            conv_relu_bn(n_unit, n_unit, 1),
            conv_relu_bn(n_unit, n_unit, 1),
            conv_relu_bn(n_unit, n_unit, 1),
            # (B, U, 1)
        )
        self.lf_mmi_head = nn.Conv1d(n_unit, n_pdf, 1)
        self.xent_head = nn.Conv1d(n_unit, n_pdf, 1)

    def forward(self, input, aux):
        hi = self.input_layer(input)
        ha = self.aux_layer(aux)
        h = hi + ha.unsqueeze(2)
        y = self.common(h)
        return self.lf_mmi_head(y), self.xent_head(y)
