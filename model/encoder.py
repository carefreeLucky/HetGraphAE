import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.GCNlayer import GraphConvolution
from layers.attention import AttentionLayer


class EncoderAtten4Layer(nn.Module):
    def __init__(self, gcn_feature, gcn_hid1, gcn_hid2, gcn_hid3):
        super(EncoderAtten4Layer, self).__init__()

        self.execEncoder1 = GraphConvolution(gcn_feature, gcn_hid1)
        self.execEncoder2 = GraphConvolution(gcn_hid1, gcn_hid2)
        self.execEncoder3 = GraphConvolution(gcn_hid2, gcn_hid3)
        self.execEncoder4 = GraphConvolution(gcn_hid3, gcn_hid3)
        self.fileEncoder1 = GraphConvolution(gcn_feature, gcn_hid1)
        self.fileEncoder2 = GraphConvolution(gcn_hid1, gcn_hid2)
        self.fileEncoder3 = GraphConvolution(gcn_hid2, gcn_hid3)
        self.fileEncoder4 = GraphConvolution(gcn_hid3, gcn_hid3)
        self.attention = AttentionLayer(gcn_hid3 + gcn_feature, gcn_hid3 + gcn_feature)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, exec_x, exec_adj, file_x, file_adj, exec_mask_adj=None, file_mask_adj=None):
        gcn_hid1 = F.relu(self.execEncoder1(exec_x, exec_adj))
        gcn_hid1 = self.drop(gcn_hid1)
        gcn_hid2 = F.relu(self.execEncoder2(gcn_hid1, exec_adj))
        gcn_hid2 = self.drop(gcn_hid2)
        gcn_hid3 = F.relu(self.execEncoder3(gcn_hid2, exec_adj))
        gcn_hid3 = self.drop(gcn_hid3)
        gcn_hid4 = self.execEncoder4(gcn_hid3, exec_adj)
        if exec_mask_adj is not None:
            z_exec = gcn_hid4[exec_mask_adj]
            exec_x = exec_x[exec_mask_adj]
        else:
            z_exec = gcn_hid4

        gcn_hid1_ = F.relu(self.fileEncoder1(file_x, file_adj))
        gcn_hid1_ = self.drop(gcn_hid1_)
        gcn_hid2_ = F.relu(self.fileEncoder2(gcn_hid1_, file_adj))
        gcn_hid2_ = self.drop(gcn_hid2_)
        gcn_hid3_ = F.relu(self.fileEncoder3(gcn_hid2_, file_adj))
        gcn_hid3_ = self.drop(gcn_hid3_)
        gcn_hid4_ = self.fileEncoder4(gcn_hid3_, file_adj)
        if file_mask_adj is not None:
            z_file = gcn_hid4_[file_mask_adj]
            file_x = file_x[file_mask_adj]
        else:
            z_file = gcn_hid4_
        z, atten_weight = self.attention(torch.cat([z_exec, exec_x], dim=1), torch.cat([z_file, file_x], dim=1), z_exec,
                                         z_file)
        return z, atten_weight, z_exec, z_file


class EncoderAtten3Layer(nn.Module):
    def __init__(self, gcn_feature, gcn_hid1, gcn_hid2, gcn_hid3):
        super(EncoderAtten3Layer, self).__init__()

        self.execEncoder1 = GraphConvolution(gcn_feature, gcn_hid1)
        self.execEncoder2 = GraphConvolution(gcn_hid1, gcn_hid2)
        self.execEncoder3 = GraphConvolution(gcn_hid2, gcn_hid3)
        self.fileEncoder1 = GraphConvolution(gcn_feature, gcn_hid1)
        self.fileEncoder2 = GraphConvolution(gcn_hid1, gcn_hid2)
        self.fileEncoder3 = GraphConvolution(gcn_hid2, gcn_hid3)
        self.attention = AttentionLayer(gcn_hid3 + gcn_feature, gcn_hid3 + gcn_feature)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, exec_x, exec_adj, file_x, file_adj, exec_mask_adj=None, file_mask_adj=None):
        gcn_hid1 = F.relu(self.execEncoder1(exec_x, exec_adj))
        gcn_hid1 = self.drop(gcn_hid1)
        gcn_hid2 = F.relu(self.execEncoder2(gcn_hid1, exec_adj))
        gcn_hid2 = self.drop(gcn_hid2)
        gcn_hid3 = self.execEncoder3(gcn_hid2, exec_adj)
        if exec_mask_adj is not None:
            z_exec = gcn_hid3[exec_mask_adj]
            exec_x = exec_x[exec_mask_adj]
        else:
            z_exec = gcn_hid3

        gcn_hid1_ = F.relu(self.fileEncoder1(file_x, file_adj))
        gcn_hid1_ = self.drop(gcn_hid1_)
        gcn_hid2_ = F.relu(self.fileEncoder2(gcn_hid1_, file_adj))
        gcn_hid2_ = self.drop(gcn_hid2_)
        gcn_hid3_ = self.fileEncoder3(gcn_hid2_, file_adj)
        if file_mask_adj is not None:
            z_file = gcn_hid3_[file_mask_adj]
            file_x = file_x[file_mask_adj]
        else:
            z_file = gcn_hid3_
        z, atten_weight = self.attention(torch.cat([z_exec, exec_x], dim=1), torch.cat([z_file, file_x], dim=1), z_exec,
                                         z_file)
        return z, atten_weight, z_exec, z_file


class EncoderAtten2Layer(nn.Module):
    def __init__(self, gcn_feature, gcn_hid1, gcn_hid2, gcn_hid3):
        super(EncoderAtten2Layer, self).__init__()

        self.execEncoder1 = GraphConvolution(gcn_feature, gcn_hid1)
        self.execEncoder2 = GraphConvolution(gcn_hid1, gcn_hid3)
        self.fileEncoder1 = GraphConvolution(gcn_feature, gcn_hid1)
        self.fileEncoder2 = GraphConvolution(gcn_hid1, gcn_hid3)
        self.attention = AttentionLayer(gcn_hid3 + gcn_feature, gcn_hid3 + gcn_feature)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, exec_x, exec_adj, file_x, file_adj, exec_mask_adj=None, file_mask_adj=None):
        gcn_hid1 = F.relu(self.execEncoder1(exec_x, exec_adj))
        gcn_hid1 = self.drop(gcn_hid1)
        gcn_hid2 = self.execEncoder2(gcn_hid1, exec_adj)
        if exec_mask_adj is not None:
            z_exec = gcn_hid2[exec_mask_adj]
            exec_x = exec_x[exec_mask_adj]
        else:
            z_exec = gcn_hid2

        gcn_hid1_ = F.relu(self.fileEncoder1(file_x, file_adj))
        gcn_hid1_ = self.drop(gcn_hid1_)
        gcn_hid2_ = self.fileEncoder2(gcn_hid1_, file_adj)
        if file_mask_adj is not None:
            z_file = gcn_hid2_[file_mask_adj]
            file_x = file_x[file_mask_adj]
        else:
            z_file = gcn_hid2_
        z, atten_weight = self.attention(torch.cat([z_exec, exec_x], dim=1), torch.cat([z_file, file_x], dim=1), z_exec,
                                         z_file)
        return z, atten_weight, z_exec, z_file


class EncoderAtten1Layer(nn.Module):
    def __init__(self, gcn_feature, gcn_hid1, gcn_hid2, gcn_hid3):
        super(EncoderAtten1Layer, self).__init__()

        self.execEncoder1 = GraphConvolution(gcn_feature, gcn_hid3)
        self.fileEncoder1 = GraphConvolution(gcn_feature, gcn_hid3)
        self.attention = AttentionLayer(gcn_hid3 + gcn_feature, gcn_hid3 + gcn_feature)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, exec_x, exec_adj, file_x, file_adj, exec_mask_adj=None, file_mask_adj=None):
        gcn_hid1 = self.execEncoder1(exec_x, exec_adj)
        if exec_mask_adj is not None:
            z_exec = gcn_hid1[exec_mask_adj]
            exec_x = exec_x[exec_mask_adj]
        else:
            z_exec = gcn_hid1
        gcn_hid1_ = self.fileEncoder1(file_x, file_adj)
        if file_mask_adj is not None:
            z_file = gcn_hid1_[file_mask_adj]
            file_x = file_x[file_mask_adj]
        else:
            z_file = gcn_hid1_
        z, atten_weight = self.attention(torch.cat([z_exec, exec_x], dim=1), torch.cat([z_file, file_x], dim=1), z_exec,
                                         z_file)
        return z, atten_weight, z_exec, z_file


class EncoderAtten5Layer(nn.Module):
    def __init__(self, gcn_feature, gcn_hid1, gcn_hid2, gcn_hid3):
        super(EncoderAtten5Layer, self).__init__()

        self.execEncoder1 = GraphConvolution(gcn_feature, gcn_hid1)
        self.execEncoder2 = GraphConvolution(gcn_hid1, gcn_hid2)
        self.execEncoder3 = GraphConvolution(gcn_hid2, gcn_hid3)
        self.execEncoder4 = GraphConvolution(gcn_hid3, gcn_hid3)
        self.execEncoder5 = GraphConvolution(gcn_hid3, gcn_hid3)
        self.fileEncoder1 = GraphConvolution(gcn_feature, gcn_hid1)
        self.fileEncoder2 = GraphConvolution(gcn_hid1, gcn_hid2)
        self.fileEncoder3 = GraphConvolution(gcn_hid2, gcn_hid3)
        self.fileEncoder4 = GraphConvolution(gcn_hid3, gcn_hid3)
        self.fileEncoder5 = GraphConvolution(gcn_hid3, gcn_hid3)
        self.attention = AttentionLayer(gcn_hid3 + gcn_feature, gcn_hid3 + gcn_feature)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, exec_x, exec_adj, file_x, file_adj, exec_mask_adj=None, file_mask_adj=None):
        gcn_hid1 = F.relu(self.execEncoder1(exec_x, exec_adj))
        gcn_hid1 = self.drop(gcn_hid1)
        gcn_hid2 = F.relu(self.execEncoder2(gcn_hid1, exec_adj))
        gcn_hid2 = self.drop(gcn_hid2)
        gcn_hid3 = F.relu(self.execEncoder3(gcn_hid2, exec_adj))
        gcn_hid3 = self.drop(gcn_hid3)
        gcn_hid4 = F.relu(self.execEncoder4(gcn_hid3, exec_adj))
        gcn_hid4 = self.drop(gcn_hid4)
        gcn_hid5 = self.execEncoder5(gcn_hid4, exec_adj)
        if exec_mask_adj is not None:
            z_exec = gcn_hid5[exec_mask_adj]
            exec_x = exec_x[exec_mask_adj]
        else:
            z_exec = gcn_hid5

        gcn_hid1_ = F.relu(self.fileEncoder1(file_x, file_adj))
        gcn_hid1_ = self.drop(gcn_hid1_)
        gcn_hid2_ = F.relu(self.fileEncoder2(gcn_hid1_, file_adj))
        gcn_hid2_ = self.drop(gcn_hid2_)
        gcn_hid3_ = F.relu(self.fileEncoder3(gcn_hid2_, file_adj))
        gcn_hid3_ = self.drop(gcn_hid3_)
        gcn_hid4_ = F.relu(self.fileEncoder4(gcn_hid3_, file_adj))
        gcn_hid4_ = self.drop(gcn_hid4_)
        gcn_hid5_ = self.fileEncoder5(gcn_hid4_, file_adj)
        if file_mask_adj is not None:
            z_file = gcn_hid5_[file_mask_adj]
            file_x = file_x[file_mask_adj]
        else:
            z_file = gcn_hid4_
        z, atten_weight = self.attention(torch.cat([z_exec, exec_x], dim=1), torch.cat([z_file, file_x], dim=1), z_exec,
                                         z_file)
        return z, atten_weight, z_exec, z_file
