import torch
import torch.nn as nn
import torch.nn.functional as F
from equivariant_diffusion.egnn_new import EGNN, GNN
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
remove_mean_batch = EnVariationalDiffusion.remove_mean_batch
import numpy as np

class CategoricalEmbedder(nn.Module):
    """
    Embeds categorical conditions such as data sources into vector representations. 
    Now the output is reshaped to (hidden_size, hidden_size) for each sample.
    """
    def __init__(self, input_size, hidden_size):
        super(CategoricalEmbedder, self).__init__()
        self.hidden_size = hidden_size
        # 将线性层的输出维度设置为 hidden_size * hidden_size
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, labels):
        # labels: 输入的向量，比如 [0, 0, 1]，通过线性变换得到嵌入
        embeddings = self.fc(labels.float())  # 转换为浮点数，因为线性层通常处理浮点数输入
        # 重新调整嵌入的形状为 [batch_size, hidden_size, hidden_size]
        embeddings = embeddings.view(embeddings.size(0), self.hidden_size)
        return embeddings

class AdaLN(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super(AdaLN, self).__init__()
        
        # 修改gamma_net和beta_net网络，加入激活函数
        self.gamma_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),  # 从 cond_dim 到 hidden_dim
            nn.SiLU(),                       # 激活函数
            nn.Linear(hidden_dim, cond_dim)  # 从 hidden_dim 返回到 cond_dim
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim)
        )

    def forward(self, x, c):
        # 输入的x和c都是n*4的矩阵
        # gamma 和 beta 的计算
        gamma = self.gamma_net(c)  
        beta = self.beta_net(c)   
        
        # 对 x 进行归一化
        mean = x.mean(dim=-1, keepdim=True) 
        std = x.std(dim=-1, keepdim=True) 
        
        x_normalized = (x - mean) / (std + 1e-6) 
        
        # 逐元素相乘并加上beta
        return gamma * x_normalized + beta

class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf,
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics',
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None, condition_vector=False):
        super().__init__()
        self.condition_vector = condition_vector
        self.mode = mode
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim

        # 初始化 CategoricalEmbedder
        self.condition_embedder = CategoricalEmbedder(input_size=3,hidden_size=joint_nf)
        
        # AdaLN 需要两个输入: h 和 c
        self.adaln = AdaLN(joint_nf, joint_nf)

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, joint_nf),
        )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, atom_nf)
        )
        residue_nf = residue_nf+2
        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, joint_nf)
        )

        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, residue_nf)
        )

        self.edge_embedding = nn.Embedding(3, self.edge_nf) \
            if self.edge_nf is not None else None
        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf

        # 修改键类型嵌入 - 接受7种键类型索引
        self.bond_embedding = nn.Embedding(7, 8)  # 7种键类型：NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC, ANY, SELF
        
        # 边特征处理网络
        if edge_embedding_dim is not None:
            self.edge_feature_net = nn.Sequential(
                nn.Linear(edge_embedding_dim + 8, 32),
                nn.SiLU(),
                nn.Linear(32, edge_embedding_dim)
            )

        if condition_time:
            dynamics_node_nf = joint_nf + 1
        else:
            print('Warning: dynamics model is _not_ conditioned on time.')
            dynamics_node_nf = joint_nf

        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=dynamics_node_nf, in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                reflection_equiv=reflection_equivariant
            )
            self.node_nf = dynamics_node_nf
            self.update_pocket_coords = update_pocket_coords

        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=dynamics_node_nf + n_dims, in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf, out_node_nf=n_dims + dynamics_node_nf,
                device=device, act_fn=act_fn, n_layers=n_layers,
                attention=attention, normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)

        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues, prompt_labels, ref_ligand=None):
        """
        模型前向传播
        
        参数:
        - xh_atoms: 配体的坐标和特征
        - xh_residues: 残基(口袋)的坐标和特征
        - t: 时间步
        - mask_atoms: 配体的批次掩码
        - mask_residues: 残基的批次掩码
        - prompt_labels: 提示标签
        - ref_ligand: 参考配体信息（包含bonds和size）
        """
        x_atoms = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()

        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()

        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(h_residues)

        if self.condition_vector:
            # 使用条件嵌入
            condition_embedding = self.condition_embedder(prompt_labels)
            h_atoms = self.adaln(h_atoms, condition_embedding)

        # 组合两种节点类型
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t对批次中所有元素相同
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t在批次维度上不同
                h_time = t[mask]
            h = torch.cat([h, h_time], dim=1)

        # 从ref_ligand提取键信息和大小，如果提供了ref_ligand
        ref_ligand_bonds = ref_ligand.get('bonds', None) if ref_ligand is not None else None
        ref_ligand_size = ref_ligand.get('size', None) if ref_ligand is not None else None

        # 获取完整图的边和边特征
        edges, edge_attr = self.get_edges(
            mask_atoms, mask_residues, 
            x_atoms, x_residues, 
            ref_ligand_bonds=ref_ligand_bonds,
            ref_ligand_size=ref_ligand_size
        )
        
        assert torch.all(mask[edges[0]] == mask[edges[1]])

        # 使用EGNN处理
        if self.mode == 'egnn_dynamics':
            update_coords_mask = None if self.update_pocket_coords \
                else torch.cat((torch.ones_like(mask_atoms),
                                torch.zeros_like(mask_residues))).unsqueeze(1)
            h_final, x_final = self.egnn(
                h, x, edges,
                update_coords_mask=update_coords_mask,
                batch_mask=mask, 
                edge_attr=edge_attr  # 使用包含键信息的边特征
            )
            vel = (x_final - x)

        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=None, edge_attr=edge_attr)
            vel = output[:, :3]
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if self.condition_time:
            # 去除表示时间的最后一个维度
            h_final = h_final[:, :-1]

        # 解码原子和残基特征
        h_final_atoms = self.atom_decoder(h_final[:len(mask_atoms)])
        h_final_residues = self.residue_decoder(h_final[len(mask_atoms):])

        if torch.any(torch.isnan(vel)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
            else:
                raise ValueError("NaN detected in EGNN output")

        if self.update_pocket_coords:
            # 对无条件联合分布，按原始代码包含这部分
            vel = remove_mean_batch(vel, mask)

        return torch.cat([vel[:len(mask_atoms)], h_final_atoms], dim=-1), \
               torch.cat([vel[len(mask_atoms):], h_final_residues], dim=-1)

    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket, ref_ligand_bonds=None, ref_ligand_size=None):
        """
        构建边并应用ref-ligand的键信息
        
        参数:
        - batch_mask_ligand: 当前配体的批次掩码
        - batch_mask_pocket: 口袋(包含ref-ligand+真实口袋)的批次掩码
        - x_ligand: 当前配体的坐标
        - x_pocket: 口袋的坐标
        - ref_ligand_bonds: 参考配体的键信息
        - ref_ligand_size: 每个批次中参考配体的原子数量
        """
        # 基本边构建
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]
        
        # 应用距离截断
        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)
        
        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)
        
        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)
        
        # 构建完整邻接矩阵和边列表
        adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
                        torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)
        edges = torch.stack(torch.where(adj), dim=0)
        
        # 处理边特征
        if self.edge_nf > 0:
            # 创建基本边类型特征
            edge_types = torch.zeros(edges.size(1), dtype=torch.long, device=edges.device)
            ligand_mask = (edges[0] < len(batch_mask_ligand)) & (edges[1] < len(batch_mask_ligand))
            pocket_mask = (edges[0] >= len(batch_mask_ligand)) & (edges[1] >= len(batch_mask_ligand))
            edge_types[ligand_mask] = 1
            edge_types[pocket_mask] = 2
            
            # 获取基本边嵌入
            edge_attr = self.edge_embedding(edge_types)
            
            # 处理ref-ligand内部键
            if ref_ligand_bonds is not None and ref_ligand_size is not None:
                # 计算每个批次中ref-ligand的起始索引
                batch_indices = torch.unique(batch_mask_pocket)
                starts = torch.zeros_like(batch_indices)
                for i, batch_idx in enumerate(batch_indices):
                    if i > 0:
                        starts[i] = starts[i-1] + ref_ligand_size[batch_indices[i-1]]
                
                # 初始化键特征
                bond_features = torch.zeros((edges.size(1), 8), device=edges.device)
                
                # 遍历所有批次
                for batch_idx in torch.unique(batch_mask_pocket):
                    # 找出当前批次的ref-ligand范围
                    b_idx = (batch_indices == batch_idx).nonzero(as_tuple=True)[0].item()
                    start_idx = starts[b_idx].item()
                    end_idx = start_idx + ref_ligand_size[batch_idx].item()
                    
                    # 找出pocket内部边中属于ref-ligand内部的边
                    ref_ligand_internal_mask = (
                        (edges[0] >= start_idx) & (edges[0] < end_idx) &
                        (edges[1] >= start_idx) & (edges[1] < end_idx) &
                        pocket_mask
                    )
                    
                    # 处理符合条件的边
                    for edge_idx in torch.where(ref_ligand_internal_mask)[0]:
                        src, dst = edges[0, edge_idx].item(), edges[1, edge_idx].item()
                        
                        # 计算相对于ref-ligand起始位置的索引
                        src_rel = src - start_idx
                        dst_rel = dst - start_idx
                        
                        # 获取键类型 - 从one-hot向量获取索引
                        # ref_ligand_bonds形状为[batch, atoms, atoms, 7]
                        if batch_idx < ref_ligand_bonds.shape[0] and src_rel < ref_ligand_bonds.shape[1] and dst_rel < ref_ligand_bonds.shape[2]:
                            # 获取键类型的one-hot向量
                            bond_onehot = ref_ligand_bonds[batch_idx, src_rel, dst_rel]
                            
                            # 从one-hot向量获取键类型索引
                            bond_type_idx = torch.argmax(bond_onehot).item()
                            
                            # 嵌入键类型
                            bond_features[edge_idx] = self.bond_embedding(torch.tensor([bond_type_idx], device=edges.device))
                
                # 合并边特征
                combined_features = torch.cat([edge_attr, bond_features], dim=1)
                edge_attr = self.edge_feature_net(combined_features)
            
            return edges, edge_attr
        
        return edges