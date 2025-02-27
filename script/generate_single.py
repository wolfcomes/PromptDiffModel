import torch
from Bio import PDB
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch
from dataset import ProcessedLigandPocketDataset
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from equivariant_diffusion.dynamics import EGNNDynamics
from equivariant_diffusion.conditional_model import ConditionalDDPM
from tqdm import tqdm
from constants import dataset_params, FLOAT_TYPE, INT_TYPE
import utils
from analysis.molecule_builder import build_molecule, process_molecule
from torch_scatter import scatter_add, scatter_mean


# 创建保存生成帧的目录
output_dir = "../data/generated_ligand"
os.makedirs(output_dir, exist_ok=True)

device = 'cuda'


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

datadir = '../data/docking_results/processed_crossdock_noH_full_temp'
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

x_dims = 3
joint_nf =64

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

atom_mapping = {0:'H', 1:'C', 2:'N', 3:'O', 4:'F', 5:'P', 6:'S', 7:'CL', 8:'BR', 9:'I', 10: 'UNK'}

def prepare_data_from_pdb(pdb_file_ligand, pdb_file_pocket, atom_mapping, device='cuda'):
    # 解析配体的 PDB 文件
    ligand_tensor = parse_pdb(pdb_file_ligand, atom_mapping)
    ref_ligand = {
        'x': ligand_tensor[:, :3].unsqueeze(0).to(device),  # Coordinates
        'one_hot': ligand_tensor[:, 3:].to(device),  # Features (atom and residue one-hot)
        'size': torch.tensor([ligand_tensor.shape[0]], dtype=torch.int64, device=device),
        'mask': torch.zeros(ligand_tensor.shape[0], dtype=torch.int64, device=device)
    }
    
    # 解析口袋的 PDB 文件
    pocket_tensor = parse_pdb(pdb_file_pocket, atom_mapping)
    pocket = {
        'x': pocket_tensor[:, :3].unsqueeze(0).to(device),  # Coordinates
        'one_hot': pocket_tensor[:, 3:].to(device),  # Features (atom and residue one-hot)
        'size': torch.tensor([pocket_tensor.shape[0]], dtype=torch.int64, device=device),
        'mask': torch.zeros(pocket_tensor.shape[0], dtype=torch.int64, device=device)
    }

    atom_num_1 = ref_ligand['one_hot'].shape[0]
    atom_num_2 = pocket['one_hot'].shape[0]
    additional_tensor_1 = torch.tensor([[1, 0]]).repeat(atom_num_1, 1).to('cuda')
    additional_tensor_2 = torch.tensor([[0, 1]]).repeat(atom_num_2, 1).to('cuda')
    ref_ligand['one_hot'] = torch.cat((ref_ligand['one_hot'], additional_tensor_1), dim=1)
    pocket['one_hot'] = torch.cat([pocket['one_hot'],additional_tensor_2],dim =1)

    return ref_ligand, pocket

def parse_pdb(pdb_file, atom_mapping):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    # Extract coordinates, atom types, and residues
    coordinates = []
    atom_types = []
    
    for atom in structure.get_atoms():
        coord = atom.coord
        atom_name = atom.element.upper()
        if atom_name == 'H':
            continue
        # Encode atom and residue types
        atom_type_idx = [key for key, value in atom_mapping.items() if value == atom_name]
        
        if not atom_type_idx:
            atom_type_idx = [10]  # UNK
        
        coordinates.append(coord)
        atom_types.append(atom_type_idx[0])
    
    # Convert to tensors
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    atom_type_one_hot = torch.nn.functional.one_hot(torch.tensor(atom_types), num_classes=len(atom_mapping)).float()
    
    # Concatenate to form the input tensor: [coord, atom_type_one_hot, residue_one_hot]
    input_tensor = torch.cat([coordinates, atom_type_one_hot], dim=1)
    
    return input_tensor

def write_pdb(generated_sdf, output_dir, batch_idx, atom_mapping):
    """保存生成的小分子为 PDB 文件"""
    pdb_lines = []
    atom_counter = 1  # PDB 文件中的原子计数

    # 从 generated_sdf 中提取配体数据
    coordinates = generated_sdf[:, :3]  # 原子坐标
    atom_type_one_hot = generated_sdf[:, 3:14]  # 原子类型 one-hot 编码

    # 解码 one-hot 向量以获取原子类型
    atom_types = atom_type_one_hot.argmax(axis=1)  # 获取原子类型索引
    lig_mask = torch.zeros(len(coordinates), dtype=torch.long)
    molecules = []
    sanitize=False
    relax_iter=0
    largest_frag=False
    for mol_pc in zip(utils.batch_to_list(coordinates, lig_mask),
                    utils.batch_to_list(atom_types, lig_mask)):

        mol = build_molecule(*mol_pc, dataset_info, add_coords=True)
        mol = process_molecule(mol,
                                add_hydrogens=False,
                                sanitize=sanitize,
                                relax_iter=relax_iter,
                                largest_frag=largest_frag)
        if mol is not None:
            molecules.append(mol)
    
    for i, mol in enumerate(molecules):
        # 创建一个输出目录，如果不存在的话
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存 SDF 文件
        sdf_file_path = os.path.join(output_dir, f"generated_molecule_{batch_idx}_{i}.sdf")
        
        # 使用 RDKit 将分子写入 SDF 格式
        writer = Chem.SDWriter(sdf_file_path)
        writer.write(mol)
        writer.close()
    
    return molecules

    # for i, (coord, atom_type_idx) in enumerate(zip(coordinates, atom_types)):
    #     atom_type_idx = int(atom_type_idx.item())  # 转换为 Python 整数
    #     atom_type = atom_mapping.get(atom_type_idx, 'UNK')  # 获取原子类型

    #     # 格式化 PDB 行
    #     pdb_line = f"HETATM{atom_counter:5d} {atom_type:>2} LIG     1    " \
    #                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           {atom_type:>2}"
    #     pdb_lines.append(pdb_line)
    #     atom_counter += 1

    # # 写入 PDB 文件
    # pdb_filename = os.path.join(output_dir, f"generated_molecule_{batch_idx}.pdb")
    # with open(pdb_filename, 'w') as pdb_file:
    #     pdb_file.write("\n".join(pdb_lines))


# 调用例子
pdb_file_ligand = '/data/home/zhangzhiyong/lead_optimization/SDE/sample/7sna_cut10_ligand.pdb'
pdb_file_pocket = '/data/home/zhangzhiyong/lead_optimization/SDE/sample/7sna_cut10_pocket.pdb'
# pdb_file_ligand = '/data/home/zhangzhiyong/lead_optimization/PromptDiffModel/data/docking_results/group_1/1b9t_A_rec_1vcj_iba_lig_tt_docked/1b9t_A_rec_1vcj_iba_lig_tt_docked_0_pocket10.pdb'
# pdb_file_pocket = '/data/home/zhangzhiyong/lead_optimization/PromptDiffModel/data/docking_results/group_1/1b9t_A_rec_1vcj_iba_lig_tt_docked/1b9t_A_rec_1vcj_iba_lig_tt_docked_generated_1_docked_poses.pdb'
ref_ligand, pocket = prepare_data_from_pdb(pdb_file_ligand, pdb_file_pocket, atom_mapping, device='cuda')
# 对字典中的每个张量应用 squeeze
ref_ligand = {key: value.squeeze(0) for key, value in ref_ligand.items()}
pocket = {key: value.squeeze(0) for key, value in pocket.items()}

pocket['mask'] = ref_ligand['mask']
pocket['x'] = ref_ligand['x']
pocket['one_hot'] = ref_ligand['one_hot']
pocket['size'] = ref_ligand['size'].unsqueeze(0)

# pocket['mask'] = torch.cat([ref_ligand['mask'],pocket['mask']],dim =0)
# pocket['x'] = torch.cat([ref_ligand['x'],pocket['x']],dim =0)
# pocket['one_hot'] = torch.cat([ref_ligand['one_hot'],pocket['one_hot']],dim =0)
# pocket['size'] = (ref_ligand['size'] + pocket['size']).unsqueeze(0)
pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

print(pocket['one_hot'].size())
print(pocket['x'].size())

prompt_labels = torch.tensor([[0,0,1]]).repeat(28, 1).to('cuda')
# pocket['mask'] = torch.tensor([0],device = device)
# pocket['x'] = torch.tensor([[0.0,0.0,0.0]],device = device)
# pocket['one_hot'] = torch.tensor([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0]],device = device)


device = 'cuda'
#checkpoint = torch.load('../checkpoints/zinc/cddpm_epoch_45.pth')
checkpoint = torch.load('../checkpoints/zinc/cddpm_ligand_only_epoch_32.pth')
#checkpoint = torch.load('../checkpoints/chembl/chembl_epoch_5.pth')
#checkpoint = torch.load('../checkpoints/cddpm/cddpm_epoch_250.pth')

cddpm.load_state_dict(checkpoint['model_state_dict'])
cddpm.to(device)
cddpm.eval()
# Generate frames using the SDE model
with torch.no_grad():
    
    cddpm.to(device)

    xh_lig, xh_pocket, lig_mask, pocket_mask = cddpm.sample_given_pocket(pocket, prompt_labels, num_nodes_lig = 28, return_frames=1,
                            timesteps=1000)

    
    pocket_com_after = scatter_mean(xh_pocket[:, :x_dims], pocket_mask, dim=0)
    xh_lig[:, :x_dims] += \
    (pocket_com_before - pocket_com_after)[lig_mask]

    write_pdb(xh_lig, output_dir,0, atom_mapping)