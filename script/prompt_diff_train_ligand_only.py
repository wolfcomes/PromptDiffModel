from pathlib import Path
import numpy as np
import pandas as pd
import h5py

import torch
from torch.utils.data import Dataset

from dataset import ProcessedLigandPocketDataset
from pathlib import Path
import torch_geometric.transforms as T
from constants import dataset_params, FLOAT_TYPE, INT_TYPE
import utils
from torch.utils.data import DataLoader
from equivariant_diffusion.dynamics import EGNNDynamics
from equivariant_diffusion.conditional_model import ConditionalDDPM
import math
import torch.nn.functional as F

datadir = '../data/docking_results/processed_crossdock_noH_full_temp'

def get_prompts(data):
    # 创建一个张量 [0, 0, 1]
    prompts = torch.tensor(data['prompt_labels']).to('cuda', INT_TYPE)
    
    return prompts

def get_ligand_and_pocket(data,virtual_nodes):
    ref_ligand = {
        'x': data['ref_lig_coords'].to('cuda', FLOAT_TYPE),
        'one_hot': data['ref_lig_one_hot'].to('cuda', FLOAT_TYPE),
        'size': data['num_ref_lig_atoms'].to('cuda', INT_TYPE),
        'mask': data['ref_lig_mask'].to('cuda', INT_TYPE),
    }
    if virtual_nodes:
        ref_ligand['num_virtual_atoms'] = data['num_virtual_atoms'].to('cuda', INT_TYPE)
    
    opt_ligand = {
        'x': data['opt_lig_coords'].to('cuda', FLOAT_TYPE),
        'one_hot': data['opt_lig_one_hot'].to('cuda', FLOAT_TYPE),
        'size': data['num_opt_lig_atoms'].to('cuda', INT_TYPE),
        'mask': data['opt_lig_mask'].to('cuda', INT_TYPE),
    }
    if virtual_nodes:
        opt_ligand['num_virtual_atoms'] = data['num_virtual_atoms'].to('cuda', INT_TYPE)

    pocket = {
        'x': data['pocket_coords'].to('cuda', FLOAT_TYPE),
        'one_hot': data['pocket_one_hot'].to('cuda', FLOAT_TYPE),
        'size': data['num_pocket_nodes'].to('cuda', INT_TYPE),
        'mask': data['pocket_mask'].to('cuda', INT_TYPE)
    }

    atom_num_1 = ref_ligand['one_hot'].shape[0]
    atom_num_2 = pocket['one_hot'].shape[0]
    additional_tensor_1 = torch.tensor([[1, 0]]).repeat(atom_num_1, 1).to('cuda')
    additional_tensor_2 = torch.tensor([[0, 1]]).repeat(atom_num_2, 1).to('cuda')
    ref_ligand['one_hot'] = torch.cat((ref_ligand['one_hot'], additional_tensor_1), dim=1)
    pocket['one_hot'] = torch.cat([pocket['one_hot'],additional_tensor_2],dim =1)

    return ref_ligand, pocket, opt_ligand

def sigma(gamma, target_tensor):
        """Computes sigma given gamma."""
        return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)),
                                        target_tensor)
    
def inflate_batch_array(array, target):
    """
    Inflates the batch array (array) with only a single axis
    (i.e. shape = (batch_size,), or possibly more empty axes
    (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
    return array.view(target_shape)

class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)

class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function.
    Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


dataset_info = dataset_params['crossdock_full']
histogram_file = Path(datadir, 'size_distribution.npy')
histogram = np.load(histogram_file).tolist()



lig_type_encoder = dataset_info['atom_encoder']
lig_type_decoder = dataset_info['atom_decoder']
pocket_type_encoder = dataset_info['aa_encoder']
pocket_type_decoder = dataset_info['aa_decoder']

virtual_nodes = False
data_transform = None
max_num_nodes = len(histogram) - 1

if virtual_nodes:
    # symbol = 'virtual'

    symbol = 'Ne'  # visualize as Neon atoms
    lig_type_encoder[symbol] = len(lig_type_encoder)
    data_transform = utils.AppendVirtualNodes(
        max_num_nodes, lig_type_encoder, symbol)
    
    virtual_atom = lig_type_encoder[symbol]
    lig_type_decoder.append(symbol)


    # Update dataset_info dictionary. This is necessary for using the
    # visualization functions.
    dataset_info['atom_encoder'] = lig_type_encoder
    dataset_info['atom_decoder'] = lig_type_decoder

atom_nf = len(lig_type_decoder)
aa_nf = len(pocket_type_decoder)


train_dataset = ProcessedLigandPocketDataset(Path(datadir, 'train.npz'), transform=data_transform)
test_dataset = ProcessedLigandPocketDataset(Path(datadir, 'test.npz'), transform=data_transform)
val_dataset = ProcessedLigandPocketDataset(Path(datadir, 'val.npz'), transform=data_transform)


train_loader = DataLoader(train_dataset, batch_size=48, num_workers=24, collate_fn=train_dataset.collate_fn, shuffle=False, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=48, num_workers=24, collate_fn=val_dataset.collate_fn, shuffle=False,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=48, num_workers=24, collate_fn=test_dataset.collate_fn, shuffle=False,pin_memory=True)




x_dims = 3
joint_nf = 64


net_dynamics = EGNNDynamics(
    atom_nf = atom_nf,
    residue_nf = aa_nf,
    n_dims = x_dims,
    joint_nf = joint_nf,
    device='cuda',
    hidden_nf= 128,
    act_fn=torch.nn.SiLU(),
    n_layers= 5,
    attention= True,
    tanh=True,
    norm_constant=1,
    inv_sublayers=1,
    sin_embedding=False,
    normalization_factor=100,
    aggregation_method= 'sum' ,
    edge_cutoff_ligand=10,
    edge_cutoff_pocket=4,
    edge_cutoff_interaction=4,
    update_pocket_coords= False,
    reflection_equivariant=True,
    edge_embedding_dim=8,
    condition_vector = True
)




cddpm = ConditionalDDPM(
            dynamics = net_dynamics,
            atom_nf = atom_nf,
            residue_nf = aa_nf,
            n_dims = x_dims,
            timesteps= 1000,
            noise_schedule = 'polynomial_2',
            noise_precision = 5.0e-4,
            loss_type = 'l2',
            norm_values = [1, 4],
            size_histogram = histogram,
            virtual_node_idx=lig_type_encoder[symbol] if virtual_nodes else None
    )


import torch
from tqdm import tqdm
import os

# 假设你已经定义了模型、优化器和其他超参数
optimizer = torch.optim.Adam(cddpm.parameters(), lr=0.001)  # 选择合适的学习率
num_epochs = 200
device = 'cuda'
save_dir = '../checkpoints/zinc'  # 模型保存的文件夹路径
loss_type = 'l2'

# 创建保存目录（如果不存在）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 添加恢复训练参数
resume_epoch = 16  # 设为None表示从头开始，设为具体数字恢复指定epoch
resume_checkpoint_path = os.path.join(save_dir, f'zinc_epoch_{resume_epoch}.pth') if resume_epoch is not None else None
# 加载检查点（如果存在）
start_epoch = 0
if resume_epoch is not None and os.path.exists(resume_checkpoint_path):
    checkpoint = torch.load(resume_checkpoint_path)
    cddpm.load_state_dict(checkpoint['model_state_dict'])
    cddpm.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
    print(f"Loaded checkpoint from epoch {resume_epoch}, resuming from epoch {start_epoch}")

# 训练循环
for epoch in range(num_epochs):
    # 设置模型为训练模式
    cddpm.train()
    cddpm.to(device)
    total_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)

    # 训练阶段
    for batch in pbar:
        # batch = {key: batch[key].to(device) for key in batch}

        optimizer.zero_grad()  # 清空梯度

        # 提取配体和口袋数据
        ref_ligand, pocket, opt_ligand = get_ligand_and_pocket(batch, virtual_nodes)
        prompt_labels = get_prompts(batch)

        pocket['mask'] = ref_ligand['mask']
        pocket['x'] = ref_ligand['x']
        pocket['one_hot'] = ref_ligand['one_hot']
        pocket['size'] = ref_ligand['size']
        
        # 计算损失，返回 (nll, info)
        delta_log_px, error_t_lig, error_t_pocket, SNR_weight, \
        loss_0_x_ligand, loss_0_x_pocket, loss_0_h, neg_log_const_0, \
        kl_prior, t_int, xh_lig_hat, info = cddpm(ref_ligand, pocket, opt_ligand, prompt_labels, return_info=True)

        if loss_type == 'l2':
            actual_ligand_size = opt_ligand['size'] - opt_ligand['num_virtual_atoms'] if virtual_nodes else opt_ligand['size']

            # normalize loss_t
            denom_lig = x_dims * actual_ligand_size + \
                        cddpm.atom_nf * opt_ligand['size']
            error_t_lig = error_t_lig / denom_lig
            denom_pocket = (x_dims + cddpm.residue_nf) * pocket['size']
            error_t_pocket = error_t_pocket / denom_pocket
            loss_t = 0.5 * (error_t_lig + error_t_pocket)

            # normalize loss_0
            loss_0_x_ligand = loss_0_x_ligand / (x_dims * actual_ligand_size)
            loss_0_x_pocket = loss_0_x_pocket / (x_dims * pocket['size'])
            loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h

        nll = loss_t + loss_0 + kl_prior

        # print("loss", nll)
        nll = nll.mean()

        # 反向传播
        nll.backward()

        # 更新参数
        optimizer.step()

        # 累加损失
        total_loss += nll.item()
        pbar.set_postfix(nll_loss=nll.item())  # 更新进度条的后缀信息

    print(f'Epoch {epoch}, Average NLL Loss: {total_loss / len(train_loader)}')

    # 验证阶段
    cddpm.eval()  # 设置模型为评估模式
    val_loss = 0
    with torch.no_grad():  # 不需要计算梯度
        for batch in val_loader:
            # batch = {key: batch[key].to(device) for key in batch}
            
            ref_ligand, pocket, opt_ligand= get_ligand_and_pocket(batch, virtual_nodes)
            prompt_labels = get_prompts(batch)

            pocket['mask'] = ref_ligand['mask']
            pocket['x'] = ref_ligand['x']
            pocket['one_hot'] = ref_ligand['one_hot']
            pocket['size'] = ref_ligand['size']

            delta_log_px, error_t_lig, error_t_pocket, SNR_weight, \
            loss_0_x_ligand, loss_0_x_pocket, loss_0_h, neg_log_const_0, \
            kl_prior, t_int, xh_lig_hat, info = cddpm(ref_ligand, pocket,opt_ligand, prompt_labels, return_info=True)

            if loss_type == 'l2':
                actual_ligand_size = opt_ligand['size'] - opt_ligand['num_virtual_atoms'] if virtual_nodes else opt_ligand['size']

                # normalize loss_t
                denom_lig = x_dims * actual_ligand_size + \
                            cddpm.atom_nf * opt_ligand['size']
                error_t_lig = error_t_lig / denom_lig
                denom_pocket = (x_dims + cddpm.residue_nf) * pocket['size']
                error_t_pocket = error_t_pocket / denom_pocket
                loss_t = 0.5 * (error_t_lig + error_t_pocket)

                # normalize loss_0
                loss_0_x_ligand = loss_0_x_ligand / (x_dims * actual_ligand_size)
                loss_0_x_pocket = loss_0_x_pocket / (x_dims * pocket['size'])
                loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h

            nll = loss_t + loss_0 + kl_prior
            nll = nll.mean()  # 将 nll 转换为标量
            val_loss += nll.item()

    print(f'Epoch {epoch}, Validation Loss: {val_loss / len(val_loader)}')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': cddpm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_loss / len(train_loader),
        'val_loss': val_loss / len(val_loader),
    }

    # 每个 epoch 后保存模型
    torch.save(checkpoint, os.path.join(save_dir, f'cddpm_ligand_only_epoch_{epoch}.pth'))

# 最终测试阶段
cddpm.eval()  # 设置模型为评估模式
test_loss = 0
with torch.no_grad():  # 不需要计算梯度
    for batch in test_loader:
        # batch = {key: batch[key].to(device) for key in batch}

        ref_ligand, pocket, opt_ligand = get_ligand_and_pocket(batch, virtual_nodes)
        prompt_labels = get_prompts(batch)

        pocket['mask'] = ref_ligand['mask']
        pocket['x'] = ref_ligand['x']
        pocket['one_hot'] = ref_ligand['one_hot']
        pocket['size'] = ref_ligand['size']
        
        delta_log_px, error_t_lig, error_t_pocket, SNR_weight, \
        loss_0_x_ligand, loss_0_x_pocket, loss_0_h, neg_log_const_0, \
        kl_prior, t_int, xh_lig_hat, info = cddpm(ref_ligand, pocket,opt_ligand, prompt_labels, return_info=True)

        if loss_type == 'l2':
            actual_ligand_size = opt_ligand['size'] - opt_ligand['num_virtual_atoms'] if virtual_nodes else opt_ligand['size']

            # normalize loss_t
            denom_lig = x_dims * actual_ligand_size + \
                        cddpm.atom_nf * opt_ligand['size']
            error_t_lig = error_t_lig / denom_lig
            denom_pocket = (x_dims + cddpm.residue_nf) * pocket['size']
            error_t_pocket = error_t_pocket / denom_pocket
            loss_t = 0.5 * (error_t_lig + error_t_pocket)

            # normalize loss_0
            loss_0_x_ligand = loss_0_x_ligand / (x_dims * actual_ligand_size)
            loss_0_x_pocket = loss_0_x_pocket / (x_dims * pocket['size'])
            loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h

        nll = loss_t + loss_0 + kl_prior
        nll = nll.mean()  # 将 nll 转换为标量
        test_loss += nll.item()

print(f'Test Loss: {test_loss / len(test_loader)}')

# 最终保存模型
torch.save(cddpm.state_dict(), os.path.join(save_dir, 'cddpm_ligand_only_final.pth'))