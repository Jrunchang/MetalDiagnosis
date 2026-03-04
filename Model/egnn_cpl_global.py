from typing import Optional, Tuple, Union, Dict, List
from functools import partial

import torch
from torch import nn, LongTensor, Tensor
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter
import e3nn
from e3nn import o3

class BaseMLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        activation: nn.Module = nn.SiLU(),
        norm: Optional[nn.Module] = None,
        residual: bool = False, 
        last_act: bool = False,
    ) -> None:
        super(BaseMLP, self).__init__()
        self.residual = residual
        if residual:
            assert output_dim == input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Identity() if norm is None else norm(hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            nn.Identity() if norm is None else norm(output_dim),
            activation if last_act else nn.Identity()
        )

    def forward(self, x):
        return x + self.mlp(x) if self.residual else self.mlp(x)

class NodeColor(nn.Module):
    def __init__(self, hidden_dim, color_type='center_radius', max_ell=6, activation=nn.SiLU()):
        super().__init__()
        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        if color_type == 'mp':
            self.mlp_msg = MLP(input_dim=hidden_dim * 2 + 1, output_dim=hidden_dim)
            self.mlp_node_feat = MLP(input_dim=hidden_dim, output_dim=hidden_dim)
        elif color_type == 'center_radius':
            self.mlp_node_feat = MLP(input_dim=1, output_dim=hidden_dim)
        elif color_type == 'tp':
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
            self.spherical_harmonics = o3.SphericalHarmonics(
                sh_irreps, normalize=True, normalization="norm"
            )
            self.tp = o3.FullyConnectedTensorProduct(sh_irreps, sh_irreps, f'{max_ell + 1}x0e',  shared_weights=False)
            
            self.mlp_sh_coff = MLP(input_dim=hidden_dim, output_dim=self.tp.weight_numel)
            self.mlp_node_feat = MLP(input_dim=max_ell + 1, output_dim=hidden_dim)
        self.color_type = color_type
        
    def forward(self, node_feat, node_pos, batch, edge_index=None, edge_attr=None):
        center = global_mean_pool(node_pos, batch)
        pos = node_pos - center[batch]
        
        if self.color_type == 'mp':
            assert edge_index is not None
            row, col = edge_index
            dist = torch.norm(node_pos[row]-node_pos[col], dim=1, keepdim=True)
            msg = torch.cat([node_feat[row], node_feat[col], dist], dim=1)
            msg = self.mlp_msg(msg)
            scalar = scatter(src=msg, index=row, dim=0, dim_size=node_feat.size(0), reduce='mean')
        elif self.color_type == 'center_radius':
            scalar = torch.norm(pos, dim=1, keepdim=True)
        elif self.color_type == 'tp':
            sh = self.spherical_harmonics(pos)
            global_sh = global_mean_pool(sh, batch)
            scalar = self.tp(sh, global_sh[batch], self.mlp_sh_coff(node_feat))
        else:
            raise NotImplementedError
            
        return self.mlp_node_feat(scalar)

class VirtualNode(nn.Module):
    def __init__(self, num_vn=4, hidden_dim=64, activation=nn.SiLU()):
        # use e3nn FullyConnectedTensorProduct produce virtual nodes.
        super().__init__()
        self.num_vn = num_vn
        self.get_vn_pos = o3.FullyConnectedTensorProduct(
            '1x1o', '1x0e', f'{num_vn}x1o', shared_weights=False
        )
        
        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        self.mlp_vec_coff = MLP(input_dim=hidden_dim, output_dim=self.get_vn_pos.weight_numel)
    
    def forward(self, node_feat, node_pos, batch):
        center = global_mean_pool(node_pos, batch)
        pos = node_pos - center[batch]
        one = torch.ones([pos.size(0), 1], device=pos.device)

        vn_pos = global_mean_pool(
            self.get_vn_pos(pos, one, self.mlp_vec_coff(node_feat)), batch
        )
        vn_pos = vn_pos.view(-1, self.num_vn, 3)
        vn_pos = vn_pos / (torch.norm(vn_pos, dim=2, keepdim=True) + 1e-3) 
        vn_pos = vn_pos.view(-1, self.num_vn * 3)
        vn_pos = vn_pos + center.repeat(1, self.num_vn)
        
        return vn_pos
    
class NodeFeatByVN(nn.Module):
    # Calculate the distance matrix from each node to all virtual nodes
    def __init__(self, num_vn=4, hidden_dim=64, activation=nn.SiLU()):
        super().__init__()
        self.num_vn = num_vn
        
        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        self.mlp_node_feat = MLP(input_dim=num_vn ** 2, output_dim=hidden_dim)
        
    def forward(self, node_feat, node_pos, vn_pos, batch):
        info_vec = node_pos.repeat(1, self.num_vn) - vn_pos[batch]
        info_vec = info_vec.view(node_pos.size(0), self.num_vn, 3)
        info_scalar = torch.cdist(info_vec, info_vec).view(node_pos.size(0), self.num_vn ** 2)
        info_scalar = info_scalar / (torch.norm(info_scalar, dim=1, keepdim=True) + 1e-3)
        
        return self.mlp_node_feat(info_scalar)
    

class EGNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        edge_attr_dim: int = 2,
        activation: nn.Module = nn.SiLU(),
        norm: Optional[nn.Module] = None,
    ) -> None:
        super(EGNNLayer, self).__init__()
        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation, norm=norm)
        self.mlp_msg = MLP(input_dim=2 * hidden_dim + edge_attr_dim + 1, output_dim=hidden_dim, last_act=True)
        self.mlp_pos = MLP(input_dim=hidden_dim, output_dim=1)
        self.mlp_node_feat = MLP(input_dim=hidden_dim + hidden_dim, output_dim=hidden_dim)
        self.mlp_vel = MLP(input_dim=hidden_dim, output_dim=1)

    def forward(
        self,
        node_feat: Tensor, 
        node_pos: Tensor, 
        node_vel: Optional[Tensor], 
        edge_index: List[LongTensor], 
        edge_attr: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        msg, diff_pos = self.Msg(edge_index, edge_attr, node_feat, node_pos)
        msg_agg, pos_agg = self.Agg(edge_index, node_feat.size(0), msg, diff_pos)
        node_feat, node_pos = self.Upd(node_feat, node_pos, node_vel, msg_agg, pos_agg)
        return node_feat, node_pos
    #calculate message
    def Msg(self, edge_index, edge_attr, node_feat, node_pos):
        row, col = edge_index
        diff_pos = node_pos[row] - node_pos[col]
        dist = torch.norm(diff_pos, p=2, dim=-1).unsqueeze(-1) ** 2
        
        msg = torch.cat([i for i in [node_feat[row], node_feat[col], edge_attr, dist] if i is not None], dim=-1)
        msg = self.mlp_msg(msg)
        diff_pos = diff_pos * self.mlp_pos(msg)

        return msg, diff_pos
    
    def Agg(self, edge_index, dim_size, msg, diff_pos):
        row, col = edge_index
        msg_agg = scatter(src=msg, index=row, dim=0, dim_size=dim_size, reduce='mean')
        pos_agg = scatter(src=diff_pos, index=row, dim=0, dim_size=dim_size, reduce='mean')
        return msg_agg, pos_agg

    def Upd(self, node_feat, node_pos, node_vel, msg_agg, pos_agg):
        node_pos = node_pos + pos_agg
        if node_vel is not None:
            node_pos = node_pos + self.mlp_vel(node_feat) * node_vel

        node_feat = torch.cat([node_feat, msg_agg], dim=-1)
        node_feat = self.mlp_node_feat(node_feat)
        return node_feat, node_pos
    
class EGNNModel(torch.nn.Module):
    def __init__(
        self,
        num_layer: int = 4,
        hidden_dim: int = 64,
        node_input_dim: int = 2,
        edge_attr_dim: int = 2,
        activation: nn.Module = nn.SiLU(),
        norm: Optional[nn.Module] = None,
        device: str = 'cpu',
    ) -> None:
        super(EGNNModel, self).__init__()
        self.embedding = nn.Linear(node_input_dim, hidden_dim)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(EGNNLayer(hidden_dim, edge_attr_dim, activation, norm))
            
        self.to(device)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        node_feat = self.embedding(data.node_feat)
        node_pos = data.node_pos
        node_vel = data.node_vel if 'node_vel' in data else None

        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        for layer in self.layers:
            node_feat, node_pos = layer(node_feat, node_pos, node_vel, edge_index, edge_attr)
            
        return node_feat, node_pos

class EGNNModel_cpl_global(torch.nn.Module):
    def __init__(
        self,
        num_layer: int = 4,
        hidden_dim: int = 64,
        node_input_dim: int = 2,
        edge_attr_dim: int = 2,
        num_vn: int = 4,
        activation: nn.Module = nn.SiLU(),
        norm: Optional[nn.Module] = None,
        device: str = 'cpu',
    ) -> None:
        super(EGNNModel_cpl_global, self).__init__()
        self.embedding = nn.Linear(node_input_dim, hidden_dim)
        self.num_vn = num_vn
        self.node_color = NodeColor(hidden_dim=hidden_dim)
        self.vn = VirtualNode(num_vn=num_vn, hidden_dim=hidden_dim)
        self.node_feat_by_vn = NodeFeatByVN(num_vn=num_vn, hidden_dim=hidden_dim)
        
        # self.com_node_feat_net = EGNNModel(num_layer=4, hidden_dim=hidden_dim, node_input_dim=node_input_dim, edge_attr_dim=edge_attr_dim)
        self.com_node_feat_net = EGNNModel(num_layer=1, hidden_dim=hidden_dim, node_input_dim=node_input_dim, edge_attr_dim=edge_attr_dim)


        self.layers = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(EGNNLayer(hidden_dim, edge_attr_dim, activation, norm))
            
        self.to(device)

    def forward(self, data: Data) -> Tensor:
        node_feat = self.embedding(data.node_feat)
        node_pos = data.node_pos
        node_vel = data.node_vel if 'node_vel' in data else None

        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None
        
        node_feat = node_feat + self.node_color(node_feat, node_pos, data.batch, edge_index)

         
        vn_pos = self.vn(node_feat, node_pos, data.batch)

       
        node_feat = node_feat+ self.node_feat_by_vn(node_feat, node_pos, vn_pos, data.batch)
        
        node_feat, _ = self.com_node_feat_net(data)
       
        for layer in self.layers:
            node_feat, node_pos = layer(node_feat, node_pos, node_vel, edge_index, edge_attr)
        return node_pos