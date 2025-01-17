import math
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from torch.nn import functional as F


def msg_func(edges):
    # s = list(edges.src['h'].size())[-1]
    # return {'m': edges.data['edge_mat'].view(-1, s, s).matmul(edges.src['h'].view(s, -1)).view(-1, s)}
    return {'m': edges.data['eh'] + edges.src['h']}


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats, edge_hidden_dim, add_on_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight_edge = nn.Parameter(torch.Tensor(edge_hidden_dim, in_feats))
        # self.weight_add_on = nn.Parameter(torch.Tensor(add_on_feats, in_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.norm = 'both'

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        stdv_edge = 1. / math.sqrt(self.weight_edge.size(1))
        # stdv_add_on = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_edge.data.uniform_(-stdv_edge, stdv_edge)
        # self.weight_add_on.data.uniform_(-stdv_add_on, stdv_add_on)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h, g):
        if self.dropout:
            h = self.dropout(h)

        g.srcdata['h'] = h
        # g.srcdata['feat'] = torch.mm(g.ndata['atom_feat'], self.weight_add_on)
        g.edata['eh'] = torch.mm(g.edata['edge_feat'], self.weight_edge)

        g.update_all(msg_func, fn.sum(msg='m', out='h'))
        h = g.dstdata['h']
        # bias

        if self.norm != 'none':
            degs = g.in_degrees().to(h.device).float().clamp(min=1)
            if self.norm == 'both':
                norm = torch.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (h.dim() - 1)
            norm = torch.reshape(norm, shp)
            h = h * norm

        h = torch.mm(h, self.weight)

        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h


class GCN(nn.Module):
    def __init__(self,
                 num_element,
                 embedding_dim,
                 hidden_dim,
                 edge_hidden_dim,
                 num_classes,
                 num_layers,
                 add_on_feats,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.element_emebdding = nn.Embedding(num_element, embedding_dim)
        self.edge_emebdding = nn.Embedding(4, edge_hidden_dim)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(embedding_dim + add_on_feats, hidden_dim, edge_hidden_dim, add_on_feats, activation, False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, edge_hidden_dim, add_on_feats, activation, dropout))
        # output layer
        self.out = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
        self.add_on = add_on_feats

    def forward(self, g):
        h = self.element_emebdding(g.ndata['elem_idx'])
        if self.add_on > 0:
            h = torch.cat((h, g.ndata['atom_feat']), dim=1)
        g.edata['edge_feat'] = self.edge_emebdding(g.edata['order'])

        for layer in self.layers:
            h = layer(h, g)

        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.out(hg)


class MLP(nn.Module):
    def __init__(self,
                 num_element,
                 embedding_dim, hidden_dim_add_on,
                 hidden_dim,
                 num_classes,
                 num_layers,
                 add_on_feats, max_len,
                 dropout):
        super(MLP, self).__init__()
        self.element_emebdding = nn.Embedding(num_element, embedding_dim, padding_idx=0)
        self.layers = nn.ModuleList()
        if add_on_feats > 0:
            self.add_on_trans = nn.Linear(add_on_feats, hidden_dim_add_on)
            self.layers.append(nn.Linear(embedding_dim + hidden_dim_add_on, hidden_dim))
        else:
            self.layers.append(nn.Linear(embedding_dim, hidden_dim))
        # input layer

        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        # output layer
        self.out = nn.Linear(hidden_dim, num_classes)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False
        self.add_on_feats = add_on_feats
        self.hidden_dim_add_on = hidden_dim_add_on
        self.max_len = max_len

    def forward(self, x, n_atoms, features):
        h = self.element_emebdding(x)
        h = torch.sum(h, dim=1) / n_atoms.unsqueeze(1)
        if self.add_on_feats > 0:
            features = features.view(-1, self.add_on_feats)
            features = self.add_on_trans(features).view(-1, self.max_len, self.hidden_dim_add_on)
            features = torch.sum(features, dim=1) / n_atoms.unsqueeze(1)

            h = torch.cat((h, features), dim=1)

        for layer in self.layers:
            h = layer(h)
            h = F.relu(h)
            if self.dropout:
                h = self.dropout(h)

        return self.out(h)


if __name__ == '__main__':
    # import torch.nn.functional as F
    # import torch.optim as optim
    # from torch.utils.data import DataLoader
    # from gcn_utils import collate
    # from utils import get_data, get_data_for_mlp
    # from tqdm import tqdm
    # #
    # # train_data = get_data('train_cv/fold_0/train.csv', device='cuda:0')
    # # dev_data = get_data('train_cv/fold_0/dev.csv', device='cuda:0')
    # # data_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate)
    # # _dev_data_loader = DataLoader(dev_data, batch_size=32, shuffle=False, collate_fn=collate)
    # #
    # # model = GCN(27, 16, 48, 2, 3, F.relu, 0.1)
    # # loss_func = nn.CrossEntropyLoss()
    # # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # #
    # # device = torch.device('cuda:0')
    # # model.to(device)
    # #
    # # for epoch in range(80):
    # #     model.train()
    # #     epoch_loss = 0
    # #     batch = tqdm(data_loader)
    # #     for bg, label in batch:
    # #         optimizer.zero_grad()
    # #         prediction = model(bg)
    # #         loss = torch.mean(F.cross_entropy(prediction, label, reduction='none')
    # #                          * (torch.ones_like(label, dtype=torch.float, device=device) +
    # #                             torch.tensor(label.clone().detach(), dtype=torch.float, device=device) * 10))
    # #
    # #         loss.backward()
    # #         optimizer.step()
    # #         epoch_loss += loss.detach().item()
    # #         batch.set_description(f'epoch {epoch} loss {epoch_loss}')
    # #     # if epoch > 10:
    # #     #     print(F.softmax(prediction.detach()).cpu())
    # #     #     print(label.cpu())
    # #
    # #     if epoch % 10 == 9:
    # #         model.eval()
    # #         epoch_loss = 0
    # #         dev_batch = tqdm(_dev_data_loader)
    # #         all_pred = []
    # #         all_label = []
    # #         with torch.no_grad():
    # #             for bg, label in dev_batch:
    # #                 prediction = model(bg)
    # #                 loss = torch.mean(F.cross_entropy(prediction, label, reduction='none')
    # #                                   * (torch.ones_like(label, dtype=torch.float, device=device) +
    # #                                      torch.tensor(label.clone().detach(), dtype=torch.float, device=device) * 10))
    # #
    # #                 all_pred.append(F.softmax(prediction.clone().detach()))
    # #                 all_label.append(label.clone().detach())
    # #
    # #                 dev_batch.set_description(f'-DEV- epoch {epoch} loss {epoch_loss}')
    # #         print(all_pred)
    # #         print(all_label)
    # # train_data = get_data('gdb_9_clean.tsv', dataset='gdb-9', device='cuda:0')
    # # data_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate)
    # # model = GCN(27, 16, 32, 11, 2, F.relu, 0.1)
    # # loss_func = nn.MSELoss(reduction='none')
    # # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # # model.train()
    # # device = torch.device('cuda:0')
    # # model.to(device)
    # #
    # # for epoch in range(80):
    # #     epoch_loss = 0
    # #     batch = tqdm(data_loader)
    # #     for bg, label in batch:
    # #         prediction = model(bg)
    # #         loss = torch.mean(loss_func(prediction, label))
    # #         optimizer.zero_grad()
    # #         loss.backward()
    # #         optimizer.step()
    # #         epoch_loss += loss.detach().item()
    # #         batch.set_description(f'epoch {epoch} loss {epoch_loss}')
    # #     if epoch > 10:
    # #         print(prediction.detach().cpu())
    # #         print(label.cpu())
    # train_data = get_data_for_mlp('train_cv/fold_0/train.csv', device='cuda:0')
    # dev_data = get_data_for_mlp('train_cv/fold_0/dev.csv', device='cuda:0')
    # data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    # _dev_data_loader = DataLoader(dev_data, batch_size=32, shuffle=False)
    #
    # model = MLP(27 + 1, 16, 48, 2, 3, 0.1)
    # loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    # device = torch.device('cuda:0')
    # model.to(device)
    #
    # for epoch in range(80):
    #     model.train()
    #     epoch_loss = 0
    #     batch = tqdm(data_loader)
    #     for elem, label, lengths in batch:
    #         optimizer.zero_grad()
    #         prediction = model(elem, lengths)
    #         loss = torch.mean(F.cross_entropy(prediction, label, reduction='none')
    #                           * (torch.ones_like(label, dtype=torch.float, device=device) +
    #                              torch.tensor(label.clone().detach(), dtype=torch.float, device=device) * 10))
    #
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.detach().item()
    #         batch.set_description(f'epoch {epoch} loss {epoch_loss}')
    #     # if epoch > 10:
    #     #     print(F.softmax(prediction.detach()).cpu())
    #     #     print(label.cpu())
    #
    #     if epoch % 10 == 9:
    #         model.eval()
    #         epoch_loss = 0
    #         dev_batch = tqdm(_dev_data_loader)
    #         all_pred = []
    #         all_label = []
    #         with torch.no_grad():
    #             for elem, label, lengths in dev_batch:
    #                 prediction = model(elem, lengths)
    #                 loss = torch.mean(F.cross_entropy(prediction, label, reduction='none')
    #                                   * (torch.ones_like(label, dtype=torch.float, device=device) +
    #                                      torch.tensor(label.clone().detach(), dtype=torch.float, device=device) * 10))
    #                 epoch_loss = loss.item()
    #                 all_pred.append(F.softmax(prediction.clone().detach()))
    #                 all_label.append(label.clone().detach())
    #
    #                 dev_batch.set_description(f'-DEV- epoch {epoch} loss {epoch_loss}')
    #         print(all_pred)
    #         print(all_label)
    NUM_ELEM = 27 + 1
    EMBEDDING_DIM = 16
    HIDDEN_DIM = 24
    HIDDEN_DIM_ADD_ON = 8
    EDGE_HIDDEN_DIM = 6
    NUM_CLS = 2
    NUM_LYS = 3
    ADD_ON_FEATS = 9  # 9

    model = MLP(NUM_ELEM, EMBEDDING_DIM, HIDDEN_DIM_ADD_ON, HIDDEN_DIM, NUM_CLS, NUM_LYS, ADD_ON_FEATS, 100,
                0.1)
    # model = GCN(NUM_ELEM, EMBEDDING_DIM, HIDDEN_DIM, EDGE_HIDDEN_DIM, NUM_CLS, NUM_LYS, ADD_ON_FEATS, F.relu,
    #             0.1)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))