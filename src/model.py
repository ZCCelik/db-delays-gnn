import torch
from torch_geometric.nn import GCNConv

class DeutscheBahnGNN(torch.nn.Module):
    
    def __init__(self, node_feat_dim, gcn_hidden, station_emb_dim, ride_feat_dim, mlp_hidden):

        super().__init__()
        self.gcn1 = GCNConv(node_feat_dim, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, station_emb_dim)
        self.head1 = torch.nn.Linear(station_emb_dim + ride_feat_dim, mlp_hidden)
        self.head2 = torch.nn.Linear(mlp_hidden, 1)
        
    def forward(self, x, edge_index, station_ids, ride_features):
        
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h= self.gcn2(h, edge_index)
        
        station_embeddings = h[station_ids]
        combined = torch.cat([station_embeddings, ride_features], axis=1 )
        
        h = self.head1(combined)
        h = torch.relu(h)
        h = self.head2(h)
        
        return h.squeeze(-1)