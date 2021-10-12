import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, gcn_feature, gcn_hid1, gcn_hid2, gcn_hid3):
        super(DNN, self).__init__()

        self.Decoder1 = nn.Linear(gcn_feature, gcn_hid1)
        self.Decoder2 = nn.Linear(gcn_hid1, gcn_hid2)
        self.Decoder3 = nn.Linear(gcn_hid2, gcn_hid3)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x):
        gcn_hid1 = F.relu(self.Decoder1(x))
        gcn_hid1 = self.drop(gcn_hid1)
        gcn_hid2 = F.relu(self.Decoder2(gcn_hid1))
        gcn_hid2 = self.drop(gcn_hid2)
        gcn_hid3 = self.Decoder3(gcn_hid2)
        return gcn_hid3
