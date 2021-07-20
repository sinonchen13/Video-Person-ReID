import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, channel, reduction=32,seq_len=4):
        print("CoordAtt")
        super(CoordAtt, self).__init__()
        self.seq_len = seq_len
        oup=channel
        inp= channel
        conv_nd = nn.Conv3d
        bn = nn.BatchNorm3d
        adaptiveavgpool = nn.AdaptiveAvgPool3d
        # t,h,w
        self.pool_t = adaptiveavgpool((None, 1, 1))
        self.pool_h = adaptiveavgpool((1, None, 1))
        self.pool_w = adaptiveavgpool((1, 1, None))
        mip = max(8, inp // reduction)

        self.conv1 = conv_nd(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = bn(mip)
        self.act = h_swish()

        self.conv_h = conv_nd(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = conv_nd(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_t = conv_nd(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        bt, c, h, w = x.size()
        b = bt//self.seq_len
        t = self.seq_len

        x = x.view(b, t, c, h, w)
        x = x.permute(0, 2, 1, 3, 4)  # b,c,t,h,w

        x_h = self.pool_h(x).permute(0, 1, 3, 2, 4)  # b,c',1,h,1->b,c',h,1,1

        x_w = self.pool_w(x).permute(0, 1, 4, 3, 2)  # b,c',1,1,w ->b,c',w,1,1
        x_t = self.pool_t(x)  # b,c',t,1,1

        y = torch.cat([x_h, x_w, x_t], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w, x_t = torch.split(y, [h, w, t], dim=2)
        x_w = x_w.permute(0, 1, 4, 3, 2)
        x_h = x_h.permute(0, 1, 3, 2, 4)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_t = self.conv_t(x_t).sigmoid()

        out = x * a_w * a_h * a_t  # b,c,t,h,w
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(bt, c, h, w)

        return out


