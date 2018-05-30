import torch
from torch import nn

def conv_relu_bn(n_in, n_out, kernel_size=3, stride=1, padding=0, dilation=1):
    """
    n_time_out = floor((n_time_in + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    """
    return nn.Sequential(
        nn.Conv1d(n_in, n_out, kernel_size, stride, padding, dilation),
        nn.BatchNorm1d(n_out, eps=1e-3),
        nn.ReLU()
    )


class SimpleTDNN(nn.Module):
    def __init__(self, n_pdf, n_freq=40, n_aux=100, n_time=29, n_stride=3, n_unit=1024, n_bottleneck=320, lda_mat=None):
        """
        total kernel width should be 29 and stride 3
        """
        super().__init__()

        # T=169
        n_input_kernel = 5
        if lda_mat is not None:
            n_first_unit = lda_mat.shape[0]
        else:
            n_first_unit = n_unit

        self.input_layer = nn.Conv1d(n_freq, n_first_unit, n_input_kernel, stride=1)
        self.aux_layer = nn.Linear(n_aux, n_first_unit)

        self.common = nn.Sequential(
            conv_relu_bn(n_first_unit, n_unit, 1),
            # 3*2=6
            conv_relu_bn(n_unit, n_unit, 3),
            conv_relu_bn(n_unit, n_unit, 1),
            # 3*(2+2)=12
            conv_relu_bn(n_unit, n_unit, 3),
            conv_relu_bn(n_unit, n_unit, 1),
            # 3*(2+2+3*2)=30
            conv_relu_bn(n_unit, n_unit, 3, 1, 0, 3),
            conv_relu_bn(n_unit, n_unit, 3, 1, 0, 3),
            conv_relu_bn(n_unit, n_unit, 3, 1, 0, 3),
            conv_relu_bn(n_unit, n_unit, 1),
            # (B, U, 1)
        )
        # T=47 (169 - 29) / 3
        self.lf_mmi_head = nn.Sequential(
            conv_relu_bn(n_unit, n_bottleneck, 3, 3),
            nn.Conv1d(n_bottleneck, n_pdf, 1)
        )
        self.xent_head = nn.Sequential(
            conv_relu_bn(n_unit, n_bottleneck, 3, 3),
            nn.Conv1d(n_bottleneck, n_pdf, 1)
        )

        self.init_weight()
        self.lda_mat = lda_mat
        self.set_lda()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # nn.init.normal_(m.weight, std=1e-5)
                # nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                # nn.init.constant_(m.bias, 0)
                # kaldi way
                nn.init.normal_(m.weight, std=1.0 / (m.weight.shape[1] ** 0.5))
                nn.init.normal_(m.bias, std=1.0)

    def kaldi_like_parameters(self):
        ng_params = [
            self.input_layer.weight,
            self.input_layer.bias,
            self.aux_layer.weight,
            self.aux_layer.bias,
            self.lf_mmi_head[-1].weight,
            self.lf_mmi_head[-1].bias,
            self.xent_head[-1].weight,
            self.xent_head[-1].bias,
        ]
        bodies = []
        for p in self.parameters():
            found = False
            for n in ng_params:
                if p is n:
                    found = True
                    break
            if not found:
                bodies.append(p)
        heads = list(self.lf_mmi_head[-1].parameters()) + list(self.xent_head[-1].parameters())
        return [{"params": bodies, "weight_decay": 0.05}, {"params": heads, "weight_decay": 0.01}]

    def set_lda(self):
        if self.lda_mat is not None:
            n_out, n_freq, n_input_kernel = self.input_layer.weight.shape
            n_ivec = self.aux_layer.weight.shape[1]
            n_inp = n_freq * n_input_kernel
            self.input_layer.weight.requires_grad = False
            self.input_layer.weight[:] = self.lda_mat[:, :n_inp].view(n_out, n_freq, n_input_kernel)
            self.input_layer.bias.requires_grad = False
            self.input_layer.bias.zero_()
            self.aux_layer.weight.requires_grad = False
            self.aux_layer.weight[:] = self.lda_mat[:, -n_ivec:]
            self.aux_layer.bias.requires_grad = False
            self.aux_layer.bias[:] = self.lda_mat[:, -1]

    def forward(self, input, aux):
        hi = self.input_layer(input)
        ha = self.aux_layer(aux)
        h = hi + ha.unsqueeze(2)
        y = self.common(h)[:, :, :-2]
        return self.lf_mmi_head(y), self.xent_head(y)
