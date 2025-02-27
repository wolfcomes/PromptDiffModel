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


def process_ligand(sdffile, atom_dict):
    try:
        ligand = Chem.SDMolSupplier(str(sdffile), sanitize=False)[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')
    if ligand is None:
        print(f"Error: Failed to load ligand from {sdffile}")
        return None

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
        return None

    lig_coords = np.array(lig_coords)

    # Generate virtual pocket (single atom at origin)
    virtual_pocket = {
        'pocket_coords': np.array([[0.0, 0.0, 0.0]]),
        'pocket_one_hot': np.eye(1, len(amino_acid_dict), len(amino_acid_dict)-1),  # 使用任意存在的编码
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
        raise KeyError(f'{e} not in atom dict ({sdffile})')

    ligand_data = {
        'lig_coords': lig_coords,
        'lig_one_hot': lig_one_hot,
        'lig_bonds': bonds_info_matrix[:len(non_h_atoms), :len(non_h_atoms)]
    }

    return ligand_data, virtual_pocket



def compute_smiles(positions, one_hot, mask):
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
    # Joint distribution of ligand's and pocket's number of nodes
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



# def saveall(filename, pdb_and_mol_ids, 
#             pocket_coords, pocket_one_hot, pocket_mask, 
#             opt_lig_coords, opt_lig_one_hot, opt_lig_bond, opt_lig_mask):
def saveall(filename, pdb_and_mol_ids, 
            pocket_coords, pocket_one_hot, pocket_mask, 
            opt_lig_coords, opt_lig_one_hot, opt_lig_mask):
    np.savez(filename,
             names=pdb_and_mol_ids,
             pocket_coords=pocket_coords,
             pocket_one_hot=pocket_one_hot,
             pocket_mask=pocket_mask,
             opt_lig_coords=opt_lig_coords,
             opt_lig_one_hot=opt_lig_one_hot,
            #  opt_lig_bond=opt_lig_bond, 
             opt_lig_mask=opt_lig_mask
             )
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdfdir', type=Path, help="Directory containing SDF files")
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    dataset_info = dataset_params['crossdock_full']
    amino_acid_dict = dataset_info['aa_encoder']
    atom_dict = dataset_info['atom_encoder']

    # 创建输出目录
    processed_dir = args.outdir if args.outdir else Path(args.sdfdir.parent, 'processed_virtual')
    processed_dir.mkdir(exist_ok=True, parents=True)

    # 获取所有SDF文件
    sdf_files = list(args.sdfdir.glob("*.sdf"))
    random.seed(args.random_seed)
    random.shuffle(sdf_files)
    
    # 划分数据集：80%训练，10%验证，10%测试
    n = len(sdf_files)
    train_files = sdf_files[:int(0.8*n)]
    val_files = sdf_files[int(0.8*n):int(0.9*n)]
    test_files = sdf_files[int(0.9*n):]

    data_split = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split in ['train', 'val', 'test']:
        opt_lig_coords = []
        opt_lig_one_hot = []
        opt_lig_bond = []
        opt_lig_mask = []
        pocket_coords = []
        pocket_one_hot = []
        pocket_mask = []
        mol_ids = []

        count = 0
        pbar = tqdm(data_split[split], desc=f'Processing {split}')
        
        for sdf_path in pbar:
            try:
                ligand_data, pocket_data = process_ligand(
                    sdf_path, atom_dict=atom_dict)
                if ligand_data is None:
                    continue
            except Exception as e:
                print(f"Error processing {sdf_path}: {str(e)}")
                continue

            # 添加opt ligand数据
            opt_lig_coords.append(ligand_data['lig_coords'])
            opt_lig_one_hot.append(ligand_data['lig_one_hot'])
            opt_lig_bond.append(ligand_data['lig_bonds'])
            opt_lig_mask.append(count * np.ones(len(ligand_data['lig_coords'])))

            # 添加虚拟pocket数据
            pocket_coords.append(pocket_data['pocket_coords'])
            pocket_one_hot.append(pocket_data['pocket_one_hot'])
            pocket_mask.append(count * np.ones(len(pocket_data['pocket_coords'])))

            mol_ids.append(sdf_path.stem)
            count += 1

        # 合并数据
        opt_lig_coords = np.concatenate(opt_lig_coords, axis=0) if opt_lig_coords else np.empty((0,3))
        opt_lig_one_hot = np.concatenate(opt_lig_one_hot, axis=0) if opt_lig_one_hot else np.empty((0,len(atom_dict)))
        # opt_lig_bond = np.concatenate(opt_lig_bond, axis=0) if opt_lig_bond else np.empty((0,100,7))
        opt_lig_mask = np.concatenate(opt_lig_mask, axis=0) if opt_lig_mask else np.empty((0,))

        pocket_coords = np.concatenate(pocket_coords, axis=0)
        pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
        pocket_mask = np.concatenate(pocket_mask, axis=0)

        # 保存数据
        saveall(processed_dir / f'{split}.npz',
                mol_ids,
                pocket_coords,
                pocket_one_hot,
                pocket_mask,
                opt_lig_coords,
                opt_lig_one_hot,
                # opt_lig_bond,
                opt_lig_mask)

        print(f"Processed {len(mol_ids)} molecules for {split} set")

    # 生成统计信息（可选）
    print("\nProcessing completed. Output saved to:", processed_dir)

    n_nodes = get_n_nodes(opt_lig_mask, pocket_mask, smooth_sigma=1.0)
    np.save(Path(processed_dir, 'size_distribution.npy'), n_nodes)