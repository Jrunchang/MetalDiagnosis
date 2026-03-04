import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import egnn_cpl_global as eg_global
import e3nn
from e3nn import o3
n_feat = 81  # node feature 
edge_feat = 7  # edge attribute 

class EGNNGlobalModel(nn.Module):
    """
    Graph classification model based on EGNNModel_cpl_global
    include:
    1. NodeColor: Position encoding based on the distance from the node position to the center of the graph
    2. VirtualNode: Virtual nodes are used to capture global information
    3. NodeFeatByVN: Enhancing node features through virtual nodes
    4. Multi layer EGNN layer
    5. Global pooling+classification header
    """
    def __init__(self, hidden_channels=64, num_layer=4, num_vn=2,seq_input_dim=15*2560):
        super(EGNNGlobalModel, self).__init__()
     
        self.embedding = nn.Linear(n_feat, hidden_channels)
        self.node_color = eg_global.NodeColor(hidden_dim=hidden_channels, color_type='center_radius')
        self.vn = eg_global.VirtualNode(num_vn=num_vn, hidden_dim=hidden_channels)
        self.node_feat_by_vn = eg_global.NodeFeatByVN(num_vn=num_vn, hidden_dim=hidden_channels)
        self.seq_fc = Linear(seq_input_dim, hidden_channels)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(eg_global.EGNNLayer(
                hidden_dim=hidden_channels, 
                edge_attr_dim=edge_feat,
                activation=nn.SiLU()
            ))
            
        self.lin1 = Linear(hidden_channels * 2, 64) 
        self.lin2 = Linear(64, 2) 
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, coords, batch, edge_index, edge_attr,seq_feat):
        """
        input:x: node feature [num_nodes, 81]
            coords:  [num_nodes, 3]
            batch: batch [num_nodes]
            edge_index:  [row_tensor, col_tensor] (list of 2 tensors)
            edge_attr:  [num_edges, 7]
        out: [batch_size, 2]
        """
        
        node_feat = self.embedding(x)  # [num_nodes, hidden_channels]
        node_pos = coords  # [num_nodes, 3]
        
        
        color_feat = self.node_color(node_feat, node_pos, batch, edge_index)
        node_feat = node_feat + color_feat
        
        vn_pos = self.vn(node_feat, node_pos, batch)  # [batch_size, num_vn * 3]
        
       
        vn_feat = self.node_feat_by_vn(node_feat, node_pos, vn_pos, batch)
        node_feat = node_feat + vn_feat
        
        node_vel = None  
        for layer in self.layers:
            node_feat, node_pos = layer(node_feat, node_pos, node_vel, edge_index, edge_attr)
        
        # 6. Global pooling 
        graph_feat = global_mean_pool(node_feat, batch)  # [batch_size, hidden_channels]
        graph_feat = self.dropout(graph_feat)

    
        batch_size = graph_feat.size(0)
        seq_flat = seq_feat.view(batch_size, -1)  # [batch_size, 15*2560]

        seq_feat_reduced = F.relu(self.seq_fc(seq_flat))  # [batch_size, 64]
        
       
        combined = torch.cat([graph_feat, seq_feat_reduced], dim=1)  # [batch_size, 128]
       
        combined = self.dropout(combined)
        out = F.relu(self.lin1(combined))  # [batch_size, 64]
        out = self.lin2(out)  # [batch_size, 2]    
        return out
