import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F


def adj_process(adj):
    adj = adj + np.eye(adj.shape[0])
    # 将大于等于1的元素置1
    adj = np.where(adj > 0, 1, 0)
    # 度矩阵
    adj = torch.tensor(adj).float()
    d = adj.sum(dim=1)
    D = torch.diag(torch.pow(d, -0.5))
    adj = D.mm(adj).mm(D)
    adj = adj.cpu()
    return adj


exec_adj_test = sp.load_npz('testData/execAdj_test.npz')
exec_adj_test = exec_adj_test.toarray()
file_adj_test = sp.load_npz('testData/fileAdj_test.npz')
file_adj_test = file_adj_test.toarray()
feature_test = np.load('testData/execFeature_test.npy')
feature_test = torch.tensor(feature_test).float()
exec_adj_test = adj_process(exec_adj_test)
file_adj_test = adj_process(file_adj_test)
# 加载模型

encoder = torch.load('save_model/encoder', map_location=torch.device('cpu'))
decoder = torch.load('save_model/decoder', map_location=torch.device('cpu'))
encoder.eval()
decoder.eval()

with torch.no_grad():
    z, atten_weight, z_exec, z_file = encoder(feature_test, exec_adj_test, feature_test, file_adj_test)
    re_feature = decoder(z)
    loss_fn = torch.nn.MSELoss(reduction='none')
    loss = loss_fn(re_feature, feature_test).mean(dim=1)
    print(loss.mean())
    loss = loss.cpu().detach().numpy()
    np.save('HetGraphAELoss', loss)
