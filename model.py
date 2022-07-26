import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class PathGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers, num_paths, path_length):
        super(PathGCN, self).__init__()
        self._dropout = dropout
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.in_act = nn.ReLU()

        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.layers = nn.ModuleList([
            PathGCNLayer(hidden_dim, num_paths, path_length)
            for _ in range(num_layers)
        ])
        self._alpha = 0.1
    
    def forward(self, input_x, paths):
        in_feats = F.dropout(input_x, p=self._dropout, training=self.training)
        in_feats = self.fc_in(in_feats)
        in_feats = self.in_act(in_feats)

        feats = in_feats
        for layer in self.layers:
            feats = layer(feats, paths, in_feats)
            feats = self._alpha * in_feats + (1 - self._alpha) * feats

        feats = F.dropout(feats, p=self._dropout, training=self.training)
        out = self.fc_out(feats)
        return out

    def setup_optimizer(self, lr, wd, lr_oc, wd_oc):
        param_list = [
            {"params": self.layers.parameters(), "lr": lr, "weight_decay": wd},
            {"params": itertools.chain(*[self.fc_in.parameters(), self.fc_out.parameters()]), "lr": lr_oc, "weight_decay": wd_oc} 
        ]
        return torch.optim.Adam(param_list)


class PathGCNLayer(nn.Module):
    def __init__(self, hidden_dim, num_path, path_length):
        super(PathGCNLayer, self).__init__()
        self._alpha = 0.1

        self.path_weight = nn.Parameter(torch.Tensor(1, path_length, hidden_dim))
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

        nn.init.xavier_normal_(self.path_weight, gain=1.414)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, feats, paths, init_feats):
        """
            feats: (num_nodes, d),
            paths: (num_path, num_nodes, path_length)
        """
        num_path = len(paths)
        results = []
        for i in range(num_path):
            path_feats = feats[paths[i]] # (num_nodes, path_length, d)
            path_feats = (path_feats * self.path_weight).sum(dim=1) # (num_nodes, d)
            results.append(path_feats)
        results = sum(results) / len(results)

        # results = self._alpha * init_feats + (1 - self._alpha) * results
        results = self.fc(results)
        results = F.relu(results)
        return results
    
    def _forward(self, feats, paths):
        _feats = feats[paths] # (num_path, num_nodes, path_length, d)
        params = self.weight.unsqueeze(0)
        _feats = _feats * params # (num_path, num_nodes, path_length, d)
        _feats = _feats.sum(dim=2) # (num_path, num_nodes, d)
        _feats = _feats.mean(dim=0) # (num_nodes, d)
        _feats = self.fc(_feats)
        _feats = F.relu(_feats)
        return _feats


         