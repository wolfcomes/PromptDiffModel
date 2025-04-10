from pathlib import Path
from time import time
import argparse
import shutil
import random

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import numpy as np

from rdkit import Chem
from scipy.ndimage import gaussian_filter
import itertools
import torch

from analysis.molecule_builder import build_molecule
from analysis.metrics import rdmol_to_smiles
import constants
from constants import covalent_radii, dataset_params


def process_ligand(sdffile, atom_dict, amino_acid_dict):
    """处理SDF文件中的分子数据。
    
    Args:
        sdffile: SDF文件路径
        atom_dict: 原子类型字典
        amino_acid_dict: 氨基酸类型字典
        
    Returns:
        tuple: (all_ligand_data, all_pocket_data) 包含所有处理后的分子数据
    """
    try:
        sdf_supplier = Chem.SDMolSupplier(str(sdffile), sanitize=False)
        total_mols = len(sdf_supplier)  # 获取总分子数
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')
    
    all_ligand_data = []
    all_pocket_data = []
    
    pbar = tqdm(enumerate(sdf_supplier), total=total_mols, desc=f'Processing {sdffile.name}')
    for mol_idx, ligand in pbar:
        if ligand is None:
            print(f"Warning: Failed to load molecule {mol_idx} from {sdffile}")
            continue

        lig_atoms = []
        lig_coords = []
        bonds_info = []
        
        # Process non-H atoms
        for idx, a in enumerate(ligand.GetAtoms()):
            if a.GetSymbol().capitalize() in atom_dict or a.GetSymbol() != 'H':
                if a.GetSymbol() != 'H':  # 排除氢原子
                    lig_atoms.append(a.GetSymbol())
                    lig_coords.append(list(ligand.GetConformer(0).GetAtomPosition(idx)))

        if len(lig_coords) == 0:
            continue

        lig_coords = np.array(lig_coords)

        # Generate virtual pocket (single atom at origin)
        virtual_pocket = {
            'pocket_coords': np.array([[0.0, 0.0, 0.0]]),
            'pocket_one_hot': np.eye(1, len(amino_acid_dict), len(amino_acid_dict)-1),
            'pocket_ids': ['VIRT:0']
        }

        # Process bonds matrix
        non_h_atoms = [idx for idx, a in enumerate(ligand.GetAtoms()) 
                       if a.GetSymbol().capitalize() in atom_dict and a.GetSymbol() != 'H']
        
        bond_type_map = {
            'NONE': [1,0,0,0,0,0,0],
            'SINGLE': [0,1,0,0,0,0,0],
            'DOUBLE': [0,0,1,0,0,0,0],
            'TRIPLE': [0,0,0,1,0,0,0],
            'AROMATIC': [0,0,0,0,1,0,0],
            'ANY': [0,0,0,0,0,1,0],
            'SELF': [0,0,0,0,0,0,1]
        }

        N = len(non_h_atoms)
        max_lig_num = 100
        bonds_info_matrix = np.zeros((N, max_lig_num, 7), dtype=int)

        # Simplified bond processing
        for i in range(N):
            bonds_info_matrix[i, i] = bond_type_map['SELF']  # 自连接

        try:
            lig_one_hot = np.stack([
                np.eye(1, len(atom_dict), atom_dict[a.capitalize()]).squeeze()
                for a in lig_atoms
            ])
        except KeyError as e:
            print(f"Warning: Skipping molecule {mol_idx} due to KeyError: {e} not in atom dict ({sdffile})")
            continue

        ligand_data = {
            'lig_coords': lig_coords,
            'lig_one_hot': lig_one_hot,
            'lig_bonds': bonds_info_matrix[:len(non_h_atoms), :len(non_h_atoms)]
        }

        all_ligand_data.append(ligand_data)
        all_pocket_data.append(virtual_pocket)

    if not all_ligand_data:
        return None
        
    return all_ligand_data, all_pocket_data


def compute_smiles(positions, one_hot, mask):
    """计算分子的SMILES表示。
    
    Args:
        positions: 原子坐标
        one_hot: 原子类型的one-hot编码
        mask: 分子掩码
        
    Returns:
        list: SMILES字符串列表
    """
    print("Computing SMILES ...")

    atom_types = np.argmax(one_hot, axis=-1)
    sections = np.where(np.diff(mask))[0] + 1
    positions = [torch.from_numpy(x) for x in np.split(positions, sections)]
    atom_types = [torch.from_numpy(x) for x in np.split(atom_types, sections)]

    mols_smiles = []
    pbar = tqdm(enumerate(zip(positions, atom_types)),
                total=len(np.unique(mask)))
    
    for i, (pos, atom_type) in pbar:
        mol = build_molecule(pos, atom_type, dataset_info)
        mol = rdmol_to_smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        pbar.set_description(f'{len(mols_smiles)}/{i + 1} successful')

    return mols_smiles


def get_n_nodes(lig_mask, pocket_mask, smooth_sigma=None):
    """计算配体和口袋节点数的联合分布。
    
    Args:
        lig_mask: 配体掩码
        pocket_mask: 口袋掩码
        smooth_sigma: 高斯平滑的sigma值
        
    Returns:
        np.ndarray: 节点数的联合分布直方图
    """
    if len(lig_mask) == 0 or len(pocket_mask) == 0:
        print("Warning: Empty masks provided to get_n_nodes")
        return np.zeros((1, 1))  # 返回1x1的零矩阵
        
    idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
    print(f"idx_lig: {len(idx_lig)}")
    print(f"idx_pocket: {len(idx_pocket)}")
    assert np.all(idx_lig == idx_pocket)

    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1,
                                np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(f'Original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0)

        print(f'Smoothed histogram: {np.count_nonzero(filtered_histogram)}/'
              f'{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled')

        joint_histogram = filtered_histogram

    return joint_histogram


def save_processed_data(filename, pdb_and_mol_ids, 
                       pocket_coords, pocket_one_hot, pocket_mask, 
                       opt_lig_coords, opt_lig_one_hot, opt_lig_mask):
    """保存处理后的数据到NPZ文件。
    
    Args:
        filename: 输出文件路径
        其他参数: 要保存的数据
    """
    np.savez(filename,
             names=pdb_and_mol_ids,
             pocket_coords=pocket_coords,
             pocket_one_hot=pocket_one_hot,
             pocket_mask=pocket_mask,
             opt_lig_coords=opt_lig_coords,
             opt_lig_one_hot=opt_lig_one_hot,
             opt_lig_mask=opt_lig_mask
             )
    return True

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process ZINC database for DiffLinker')
    parser.add_argument('--sdfdir', type=Path, required=True,
                       help="Directory containing SDF files")
    parser.add_argument('--outdir', type=Path, default=None,
                       help="Output directory for processed files")
    args = parser.parse_args()

    # 初始化数据集参数
    dataset_info = dataset_params['crossdock_full']
    amino_acid_dict = dataset_info['aa_encoder']
    atom_dict = dataset_info['atom_encoder']

    # 创建输出目录
    processed_dir = args.outdir if args.outdir else Path(args.sdfdir.parent, 'processed_virtual')
    processed_dir.mkdir(exist_ok=True, parents=True)

    # 固定的数据集文件名
    data_split = {
        'train': [args.sdfdir / 'zinc_final_train_mol.sdf'],
        'val': [args.sdfdir / 'zinc_final_val_mol.sdf'],
        'test': [args.sdfdir / 'zinc_final_test_mol.sdf']
    }

    # 检查文件是否存在
    for split, files in data_split.items():
        for file in files:
            if not file.exists():
                raise FileNotFoundError(f"Cannot find {split} file: {file}")

    # 处理每个数据集分割
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} set:")
        opt_lig_coords = []
        opt_lig_one_hot = []
        opt_lig_mask = []
        pocket_coords = []
        pocket_one_hot = []
        pocket_mask = []
        mol_ids = []

        count = 0
        for sdf_path in data_split[split]:
            try:
                result = process_ligand(sdf_path, atom_dict=atom_dict, amino_acid_dict=amino_acid_dict)
                if result is None:
                    continue
                    
                all_ligand_data, all_pocket_data = result
                
                # 处理每个分子
                for mol_idx, (ligand_data, pocket_data) in enumerate(zip(all_ligand_data, all_pocket_data)):
                    # 添加配体数据
                    opt_lig_coords.append(ligand_data['lig_coords'])
                    opt_lig_one_hot.append(ligand_data['lig_one_hot'])
                    opt_lig_mask.append(count * np.ones(len(ligand_data['lig_coords'])))

                    # 添加口袋数据
                    pocket_coords.append(pocket_data['pocket_coords'])
                    pocket_one_hot.append(pocket_data['pocket_one_hot'])
                    pocket_mask.append(count * np.ones(len(pocket_data['pocket_coords'])))

                    mol_ids.append(f"{sdf_path.stem}_{mol_idx}")
                    count += 1
                    
            except Exception as e:
                print(f"Error processing {sdf_path}: {str(e)}")
                continue

        # 合并数据
        if opt_lig_coords:
            opt_lig_coords = np.concatenate(opt_lig_coords, axis=0)
            opt_lig_one_hot = np.concatenate(opt_lig_one_hot, axis=0)
            opt_lig_mask = np.concatenate(opt_lig_mask, axis=0)
            pocket_coords = np.concatenate(pocket_coords, axis=0)
            pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
            pocket_mask = np.concatenate(pocket_mask, axis=0)
        else:
            # 处理空数据情况
            opt_lig_coords = np.empty((0,3))
            opt_lig_one_hot = np.empty((0,len(atom_dict)))
            opt_lig_mask = np.empty((0,))
            pocket_coords = np.empty((0,3))
            pocket_one_hot = np.empty((0,len(amino_acid_dict)))
            pocket_mask = np.empty((0,))

        # 保存数据
        save_processed_data(
            processed_dir / f'{split}.npz',
            mol_ids,
            pocket_coords,
            pocket_one_hot,
            pocket_mask,
            opt_lig_coords,
            opt_lig_one_hot,
            opt_lig_mask
        )

        print(f"Processed {len(mol_ids)} molecules for {split} set")

    print("\nProcessing completed. Output saved to:", processed_dir)

    # 生成和保存节点数分布
    n_nodes = get_n_nodes(opt_lig_mask, pocket_mask, smooth_sigma=1.0)
    np.save(Path(processed_dir, 'size_distribution.npy'), n_nodes)
