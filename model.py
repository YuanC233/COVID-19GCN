import math
import torch
import torch.nn as nn
import dgl
import dgl.function as fn


def msg_func(edges):
    return {'m': edges.data['weight'].unsqueeze(1) * edges.src['h']}


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
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
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h, g):
        if self.dropout:
            h = self.dropout(h)

        g.srcdata['h'] = h
        g.update_all(msg_func, fn.sum(msg='m', out='h'))
        h = g.dstdata['h']
        # bias

        h = torch.mm(h, self.weight)
        if self.norm != 'none':
            degs = g.in_degrees().to(h.device).float().clamp(min=1)
            if self.norm == 'both':
                norm = torch.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (h.dim() - 1)
            norm = torch.reshape(norm, shp)
            h = h * norm

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
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.element_emebdding = nn.Embedding(num_element, embedding_dim)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(embedding_dim, hidden_dim, activation, False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, activation, dropout))
        # output layer
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, g):
        h = self.element_emebdding(g.ndata['elem_idx'])
        for layer in self.layers:
            h = layer(h, g)
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.out(hg)


if __name__ == '__main__':
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from gcn_utils import collate
    from utils import get_data
    from tqdm import tqdm

    # train_data = get_data('train.csv', device='cuda:0')
    # data_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate)
    #
    # model = GCN(27, 16, 32, 2, 2, F.relu, 0.1)
    # loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # model.train()
    # device = torch.device('cuda:0')
    # model.to(device)
    #
    # for epoch in range(80):
    #     epoch_loss = 0
    #     batch = tqdm(data_loader)
    #     for bg, label in batch:
    #         prediction = model(bg)
    #         loss = torch.sum(F.cross_entropy(prediction, label, reduction='none')
    #                          * (torch.ones_like(label, dtype=torch.float, device=device) +
    #                             torch.tensor(label.clone().detach(), dtype=torch.float, device=device) * 10))
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.detach().item()
    #         batch.set_description(f'epoch {epoch} loss {epoch_loss}')
    #     if epoch > 10:
    #         print(F.softmax(prediction.detach()).cpu())
    #         print(label.cpu())

    train_data = get_data('gdb_9_clean.tsv', dataset='gdb-9', device='cuda:0')
    data_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate)
    model = GCN(27, 16, 32, 11, 2, F.relu, 0.1)
    loss_func = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    device = torch.device('cuda:0')
    model.to(device)

    for epoch in range(80):
        epoch_loss = 0
        batch = tqdm(data_loader)
        for bg, label in batch:
            prediction = model(bg)
            loss = torch.sum(loss_func(prediction, label) * 1/label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            batch.set_description(f'epoch {epoch} loss {epoch_loss}')
        if epoch > 10:
            print(prediction.detach().cpu())
            print(label.cpu())