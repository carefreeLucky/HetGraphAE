import torch.nn as nn
import torch.nn.functional as F
import torch


class AttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, drop_rt=0.4):
        super(AttentionLayerNEW, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop_layer = nn.Dropout(p=drop_rt)
        self.linear = nn.Linear(in_features, out_features)
        self.q = nn.Linear(out_features, 1)
        self.act = nn.PReLU()
        self.norm = nn.BatchNorm1d(2)

    def forward(self, exec_input, file_input, exec_hid, file_hid):
        exec_input__ = exec_input.view(exec_input.shape[0], 1, exec_input.shape[1])
        file_input__ = file_input.view(file_input.shape[0], 1, file_input.shape[1])
        inputs = torch.cat((exec_input__, file_input__), dim=1)
        hidden = self.linear(inputs)
        hidden = self.act(hidden)
        atten = self.q(hidden)
        atten = self.norm(atten)
        atten = torch.tanh(atten)
        atten = F.softmax(atten, dim=1)
        atten_cof = atten.reshape(atten.shape[0], 1, atten.shape[1])
        print('attention weight', atten_cof)
        exec_hid = exec_hid.view(exec_hid.shape[0], 1, exec_hid.shape[1])
        file_hid = file_hid.view(file_hid.shape[0], 1, file_hid.shape[1])
        hid_inputs = torch.cat((exec_hid, file_hid), dim=1)
        fuse_embed = torch.matmul(atten_cof, hid_inputs)
        fuse_embed.squeeze_()
        fuse_embed = self.drop_layer(fuse_embed)
        return fuse_embed, atten_cof
