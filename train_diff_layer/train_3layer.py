import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import scipy.sparse as sp
from construct_sub_graph import sub_graph
from model.encoder import EncoderAtten3Layer
from model.decoder_feature import DNN as DecoderFeature

print('开始训练')
execAdj = sp.load_npz('trainData/execAdj.npz')
execAdj = execAdj.tocsr()
fileAdj = sp.load_npz('trainData/fileAdj.npz')
fileAdj = fileAdj.tocsr()
feature = np.load('trainData/execFeature.npy')
exec_graph = sub_graph(execAdj, feature)
file_graph = sub_graph(fileAdj, feature)
het_adj = execAdj + fileAdj
het_graph = sub_graph(het_adj, feature)
feature_dim = feature.shape[1]
node_num = feature.shape[0]
file_sum = fileAdj.sum(axis=1)
file_nodes = []
for i in range(len(file_sum)):
    if file_sum[i][0] != 0:
        file_nodes.append(i)


class Train:

    def __init__(self, gcn_h1_dim, gcn_h2_dim, gcn_h3_dim, learn_rate1=0.001, learn_rate2=0.001, weight_decay=0.001):
        self.encoder = EncoderAtten3Layer(feature_dim, gcn_h1_dim, gcn_h2_dim, gcn_h3_dim)
        self.decoder = DecoderFeature(gcn_h3_dim, gcn_h2_dim, gcn_h1_dim, feature_dim)
        self.loss_fn_feature = torch.nn.MSELoss(reduction='mean')
        self.loss_fn_adj = torch.nn.MSELoss(reduction='none')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            self.decoder = self.decoder.to(device)
            self.encoder = self.encoder.to(device)
        self.optimizer_encoder = optim.Adam(
            [{'params': self.encoder.parameters(), 'lr': learn_rate1}],
            weight_decay=weight_decay)
        self.optimizer_decoder = optim.Adam(
            [{'params': self.decoder.parameters(), 'lr': learn_rate2}],
            weight_decay=weight_decay)

    def get_embedding(self, node):
        exec_adj, exec_feature, _, _, _, exec_mask = exec_graph.construct(node, 1)
        file_adj, file_feature, _, _, _, file_mask = file_graph.construct(node, 1)
        if torch.cuda.is_available():
            exec_adj = exec_adj.cuda()
            exec_feature = exec_feature.cuda()
            file_adj = file_adj.cuda()
            file_feature = file_feature.cuda()
        z, _, _, _ = self.encoder(exec_feature, exec_adj, file_feature, file_adj, exec_mask, file_mask)
        return z

    def batch_loss_adj(self, feature, raw_adj, re_adj):
        feature_norm = F.normalize(feature, p=2, dim=1)
        feature_sim = feature_norm @ feature_norm.t()
        if torch.cuda.is_available():
            sim_mask = torch.where(feature_sim > 0.8, torch.tensor([0]).cuda(), torch.tensor([1]).cuda())
        else:
            sim_mask = torch.where(feature_sim > 0.8, torch.tensor([0]), torch.tensor([1]))
        sim_mask = sim_mask.float()
        sim_mask += raw_adj
        if torch.cuda.is_available():
            sim_mask = torch.where(sim_mask > 0, torch.tensor([1]).cuda(), torch.tensor([0]).cuda())
        else:
            sim_mask = torch.where(sim_mask > 0, torch.tensor([1]), torch.tensor([0]))

        adj_loss = self.loss_fn_adj(re_adj * sim_mask, raw_adj).sum() / sim_mask.sum()
        return adj_loss

    def batch_loss(self, ids, a=0.05, b=0.05):
        ids = list(set(ids))
        ids.sort()
        _, het_feature, _, raw_het_adj, nodes, _ = het_graph.construct(ids, 1)
        if torch.cuda.is_available():
            het_feature = het_feature.cuda()
            raw_het_adj = raw_het_adj.cuda()
        z = self.get_embedding(nodes)
        re_feature = self.decoder(z)
        z = F.normalize(z, p=2, dim=1)
        re_adj = z @ z.t()
        re_het_adj = (re_adj + 1) / 2
        feature_loss = self.loss_fn_feature(re_feature, het_feature)
        adj_loss = self.batch_loss_adj(het_feature, raw_het_adj, re_het_adj)
        return feature_loss, adj_loss

    def train(self, batch_size=100, t=1000):
        node_list = list(range(node_num))
        random.shuffle(node_list)
        random.shuffle(file_nodes)
        start = 0
        file_start = 0
        data_set = []
        while start < (node_num - batch_size):
            if file_start > (len(file_nodes) - batch_size):
                data_set.append(file_nodes[file_start:])
                file_start = 0
            else:
                data_set.append(file_nodes[file_start:file_start + batch_size])
                file_start += batch_size
            for _ in range(6):
                if start >= (node_num - batch_size):
                    break
                data_set.append(node_list[start: start + batch_size])
                start += batch_size
        if start < node_num:
            data_set.append(node_list[start:])
        try:
            count = 0
            best = 0
            for times in range(t):
                self.encoder.train()
                self.decoder.train()
                for i in range(len(data_set)):
                    count += 1
                    print("epoch:%s, batch:%s" % (times, i))
                    loss_fea, loss_adj = self.batch_loss(data_set[i])
                    self.optimizer_encoder.zero_grad()
                    loss_adj.backward(retain_graph=True)
                    self.optimizer_encoder.step()
                    self.optimizer_decoder.zero_grad()
                    self.optimizer_encoder.zero_grad()
                    loss_fea.backward()
                    self.optimizer_decoder.step()
                    if count == 100:
                        torch.save(self.decoder, 'save_model/decoder' + str(times))
                        torch.save(self.encoder, 'save_model/encoder' + str(times))
                        count = 0
        except KeyboardInterrupt or MemoryError or RuntimeError:
            torch.save(self.decoder, 'save_model/decoder')
            torch.save(self.encoder, 'save_model/encoder')
        return self.decoder, self.encoder


SEED = 5000
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
train_ = Train(100, 90, 80, 0.001, 0.001, 0.000)
decoder, encoder = train_.train(batch_size=8, t=10)
torch.save(decoder, 'save_model/decoder')
torch.save(encoder, 'save_model/encoder')
