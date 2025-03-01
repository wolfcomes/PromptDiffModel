from pathlib import Path
from time import time
import argparse
import shutil
import random

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
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
        
    lig_atoms = []
    lig_coords = []
    atom_mapping = {}
 
    for idx, a in enumerate(ligand.GetAtoms()):
        if (a.GetSymbol().capitalize() in atom_dict or a.GetSymbol() != 'H'):
            lig_atoms.append(a.GetSymbol())
            atom_mapping[len(lig_atoms) - 1] = idx
            if a.GetSymbol() != 'H':
                lig_coords.append(list(ligand.GetConformer(0).GetAtomPosition(idx)))

    lig_coords = np.array(lig_coords)

    non_h_atoms = []
    for idx, a in enumerate(ligand.GetAtoms()):
        atom_symbol = a.GetSymbol().capitalize()
        if atom_symbol != 'H':
            non_h_atoms.append({
                'idx': idx,
                'symbol': atom_symbol
            })


    non_h_atoms = []
    for idx, a in enumerate(ligand.GetAtoms()):
        atom_symbol = a.GetSymbol().capitalize()
        if atom_symbol != 'H':  # 只选择非氢原子
            non_h_atoms.append({
                'idx': idx,
                'symbol': atom_symbol
            })

    
    bond_type_map = {
        'NONE': [1, 0, 0, 0, 0, 0, 0],  # NONE
        'SINGLE': [0, 1, 0, 0, 0, 0, 0],  # SINGLE
        'DOUBLE': [0, 0, 1, 0, 0, 0, 0],  # DOUBLE
        'TRIPLE': [0, 0, 0, 1, 0, 0, 0],  # TRIPLE
        'AROMATIC': [0, 0, 0, 0, 1, 0, 0],  # AROMATIC
        'ANY': [0, 0, 0, 0, 0, 1, 0],  # ANY (未知类型)
        'SELF': [0, 0, 0, 0, 0, 0, 1]
    }

    N = len(non_h_atoms)
    max_lig_num = 100
    bonds_info_matrix = np.zeros((N, max_lig_num, 7), dtype=int)

    # 对非氢原子进行两两配对
    for i in range(N):
        for j in range(i + 1, N):  # 只考虑 i < j 组合，避免重复
            atom1_idx = non_h_atoms[i]['idx']
            atom2_idx = non_h_atoms[j]['idx']
            
            # 检查是否有键连接
            bond_found = False
            for bond in ligand.GetBonds():
                if (bond.GetBeginAtomIdx() == atom1_idx and bond.GetEndAtomIdx() == atom2_idx) or \
                (bond.GetBeginAtomIdx() == atom2_idx and bond.GetEndAtomIdx() == atom1_idx):
                    # 找到连接的键，记录键的类型
                    bond_type_num = bond_type_map.get(bond.GetBondType().name, bond_type_map['ANY'])
                    bonds_info_matrix[i, j] = bond_type_num
                    bonds_info_matrix[j, i] = bond_type_num  # 填充对称位置
                    bond_found = True
                    break
            
            # 如果没有找到键，记录为 'NONE'
            if not bond_found:
                bonds_info_matrix[i, j] = bond_type_map['NONE']
                bonds_info_matrix[j, i] = bond_type_map['NONE']  # 双向填充

    # 处理自连接（如果有的话）
    for k in range(len(non_h_atoms)):
        bonds_info_matrix[k, k] = bond_type_map['SELF']

    for i in range(N):
        for j in range(i + 1, max_lig_num):
            if np.array_equal(bonds_info_matrix[i, j], np.array([0, 0, 0, 0, 0, 0, 0])):
                bonds_info_matrix[i, j] = bond_type_map['NONE']


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
        'lig_bonds': bonds_info_matrix
    }
    
    return ligand_data

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


def get_bond_length_arrays(atom_mapping):
    bond_arrays = []
    for i in range(3):
        bond_dict = getattr(constants, f'bonds{i + 1}')
        bond_array = np.zeros((len(atom_mapping), len(atom_mapping)))
        for a1 in atom_mapping.keys():
            for a2 in atom_mapping.keys():
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    bond_len = bond_dict[a1][a2]
                else:
                    bond_len = 0
                bond_array[atom_mapping[a1], atom_mapping[a2]] = bond_len

        assert np.all(bond_array == bond_array.T)
        bond_arrays.append(bond_array)

    return bond_arrays


def get_lennard_jones_rm(atom_mapping):
    # Bond radii for the Lennard-Jones potential
    LJ_rm = np.zeros((len(atom_mapping), len(atom_mapping)))

    for a1 in atom_mapping.keys():
        for a2 in atom_mapping.keys():
            all_bond_lengths = []
            for btype in ['bonds1', 'bonds2', 'bonds3']:
                bond_dict = getattr(constants, btype)
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    all_bond_lengths.append(bond_dict[a1][a2])

            if len(all_bond_lengths) > 0:
                # take the shortest possible bond length because slightly larger
                # values aren't penalized as much
                bond_len = min(all_bond_lengths)
            else:
                if a1 == 'others' or a2 == 'others':
                    bond_len = 0
                else:
                    # Replace missing values with sum of average covalent radii
                    bond_len = covalent_radii[a1] + covalent_radii[a2]

            LJ_rm[atom_mapping[a1], atom_mapping[a2]] = bond_len

    assert np.all(LJ_rm == LJ_rm.T)
    return LJ_rm


def get_type_histograms(lig_one_hot, pocket_one_hot, atom_encoder, aa_encoder):
    atom_decoder = list(atom_encoder.keys())
    atom_counts = {k: 0 for k in atom_encoder.keys()}
    for a in [atom_decoder[x] for x in lig_one_hot.argmax(1)]:
        atom_counts[a] += 1

    aa_decoder = list(aa_encoder.keys())
    aa_counts = {k: 0 for k in aa_encoder.keys()}
    for r in [aa_decoder[x] for x in pocket_one_hot.argmax(1)]:
        aa_counts[r] += 1

    return atom_counts, aa_counts


def saveall(filename, mol_ids, ref_lig_coords, ref_lig_one_hot, ref_lig_bonds, ref_lig_mask,
            pocket_coords, pocket_one_hot, pocket_mask,
            prompt_labels, opt_lig_coords, opt_lig_one_hot, opt_lig_bond, opt_lig_mask):
    np.savez(filename,
             names=mol_ids,
             prompt_labels=prompt_labels,
             ref_lig_coords=ref_lig_coords,
             ref_lig_one_hot=ref_lig_one_hot,
             ref_lig_bonds=ref_lig_bonds,
             ref_lig_mask=ref_lig_mask,
             opt_lig_coords=opt_lig_coords,
             opt_lig_one_hot=opt_lig_one_hot,
             opt_lig_bond=opt_lig_bond, 
             opt_lig_mask=opt_lig_mask,
             pocket_coords=pocket_coords,
             pocket_one_hot=pocket_one_hot,
             pocket_mask=pocket_mask
             )
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_dir', type=Path, help='Directory containing reference molecules')
    parser.add_argument('--opt_dir', type=Path, help='Directory containing optimized molecules')
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size ratio')
    parser.add_argument('--test_size', type=float, default=0.1, help='Test set size ratio')
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    dataset_info = dataset_params['crossdock_full']
    atom_dict = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']
    aa_encoder = dataset_info['aa_encoder']

    # Make output directory
    if args.outdir is None:
        processed_dir = Path('processed_ligands')
    else:
        processed_dir = args.outdir

    processed_dir.mkdir(exist_ok=True, parents=True)

    # 用于存储所有处理好的数据
    all_data = {
        'ref_lig_coords': [],
        'ref_lig_one_hot': [],
        'ref_lig_bond': [],
        'ref_lig_mask': [],
        'opt_lig_coords': [],
        'opt_lig_one_hot': [],
        'opt_lig_bond': [],
        'opt_lig_mask': [],
        'pocket_coords': [],
        'pocket_one_hot': [],
        'pocket_mask': [],
        'mol_ids': []
    }
    
    count = 0
    
    # 记录失败的情况
    failed_pairs = []
    failed_ref_process = []
    failed_opt_process = []

    # 获取所有子目录
    ref_subdirs = [d for d in args.ref_dir.iterdir() if d.is_dir()]
    print(f"Found {len(ref_subdirs)} subdirectories")
    
    # 处理所有分子数据
    for ref_subdir in ref_subdirs:
        opt_subdir = args.opt_dir / ref_subdir.name
        if not opt_subdir.exists():
            print(f"Warning: No matching optimized directory for {ref_subdir}")
            continue

        ref_files = list(ref_subdir.glob('*.sdf'))
        print(f"\nProcessing subdirectory {ref_subdir.name}: {len(ref_files)} reference molecules")
        
        for ref_file in tqdm(ref_files):
            try:
                try:
                    ref_data = process_ligand(ref_file, atom_dict)
                except Exception as e:
                    failed_ref_process.append((str(ref_file), str(e)))
                    print(f"Error processing reference molecule {ref_file}: {e}")
                    continue

                # 查找所有对应的优化后分子
                opt_files = list(opt_subdir.glob(f"*.sdf"))
                
                if not opt_files:
                    failed_pairs.append(str(ref_file))
                    print(f"Warning: No optimized molecules found for {ref_file}")
                    continue

                for opt_file in opt_files:
                    try:
                        opt_data = process_ligand(opt_file, atom_dict)
                    except Exception as e:
                        failed_opt_process.append((str(opt_file), str(e)))
                        print(f"Error processing optimized molecule {opt_file}: {e}")
                        continue

                    all_data['mol_ids'].append(f"{ref_subdir.name}/{ref_file.stem}->{opt_file.stem}")
                    all_data['ref_lig_coords'].append(ref_data['lig_coords'])
                    all_data['ref_lig_one_hot'].append(ref_data['lig_one_hot'])
                    all_data['ref_lig_bond'].append(ref_data['lig_bonds'])
                    all_data['ref_lig_mask'].append(count * np.ones(len(ref_data['lig_coords'])))
                    all_data['opt_lig_coords'].append(opt_data['lig_coords'])
                    all_data['opt_lig_one_hot'].append(opt_data['lig_one_hot'])
                    all_data['opt_lig_bond'].append(opt_data['lig_bonds'])
                    all_data['opt_lig_mask'].append(count * np.ones(len(opt_data['lig_coords'])))
                    all_data['pocket_coords'].append(np.array([[0.0, 0.0, 0.0]]))
                    all_data['pocket_one_hot'].append(np.eye(1, len(aa_encoder), len(aa_encoder)-1))
                    all_data['pocket_mask'].append(count * np.ones(1))
                    count += 1
                
            except Exception as e:
                print(f"Unexpected error processing {ref_file}: {e}")
                continue

    print(f"\nProcessing Summary:")
    print(f"Total subdirectories processed: {len(ref_subdirs)}")
    print(f"Successfully processed pairs: {count}")
    print(f"Failed to find optimized pairs: {len(failed_pairs)}")
    print(f"Failed to process reference molecules: {len(failed_ref_process)}")
    print(f"Failed to process optimized molecules: {len(failed_opt_process)}")

    if count == 0:
        print("No valid molecule pairs were processed. Exiting...")
        exit(1)

    # 划分数据集
    try:
        indices = list(range(count))
        random.shuffle(indices)
        
        test_count = int(count * args.test_size)
        val_count = int(count * args.val_size)
        train_count = count - test_count - val_count
        
        splits = {
            'train': indices[:train_count],
            'val': indices[train_count:train_count + val_count],
            'test': indices[train_count + val_count:]
        }
        
        print(f"\nDataset split:")
        print(f"Train set: {len(splits['train'])} samples")
        print(f"Validation set: {len(splits['val'])} samples")
        print(f"Test set: {len(splits['test'])} samples")
        
        # 保存每个数据集
        for split_name, split_indices in splits.items():
            if len(split_indices) == 0:
                continue
                
            split_data = {
                'mol_ids': [all_data['mol_ids'][i] for i in split_indices],
                'ref_lig_coords': np.concatenate([all_data['ref_lig_coords'][i] for i in split_indices], axis=0),
                'ref_lig_one_hot': np.concatenate([all_data['ref_lig_one_hot'][i] for i in split_indices], axis=0),
                'ref_lig_bond': np.concatenate([all_data['ref_lig_bond'][i] for i in split_indices], axis=0),
                'ref_lig_mask': np.concatenate([all_data['ref_lig_mask'][i] for i in split_indices], axis=0),
                'opt_lig_coords': np.concatenate([all_data['opt_lig_coords'][i] for i in split_indices], axis=0),
                'opt_lig_one_hot': np.concatenate([all_data['opt_lig_one_hot'][i] for i in split_indices], axis=0),
                'opt_lig_bond': np.concatenate([all_data['opt_lig_bond'][i] for i in split_indices], axis=0),
                'opt_lig_mask': np.concatenate([all_data['opt_lig_mask'][i] for i in split_indices], axis=0),
                'pocket_coords': np.concatenate([all_data['pocket_coords'][i] for i in split_indices], axis=0),
                'pocket_one_hot': np.concatenate([all_data['pocket_one_hot'][i] for i in split_indices], axis=0),
                'pocket_mask': np.concatenate([all_data['pocket_mask'][i] for i in split_indices], axis=0),
            }
            
            prompt_labels = np.tile([0, 0, 1], (len(split_data['opt_lig_coords']), 1))
            
            # 保存数据
            saveall(processed_dir / f'{split_name}.npz', 
                    split_data['mol_ids'], 
                    split_data['ref_lig_coords'],
                    split_data['ref_lig_one_hot'],
                    split_data['ref_lig_bond'], 
                    split_data['ref_lig_mask'],
                    split_data['pocket_coords'],
                    split_data['pocket_one_hot'],
                    split_data['pocket_mask'],
                    prompt_labels,
                    split_data['opt_lig_coords'],
                    split_data['opt_lig_one_hot'],
                    split_data['opt_lig_bond'], 
                    split_data['opt_lig_mask'])
                
        # 保存失败记录
        with open(processed_dir / 'processing_failures.txt', 'w') as f:
            f.write("Missing optimized pairs:\n")
            for pair in failed_pairs:
                f.write(f"{pair}\n")
            f.write("\nFailed reference molecules:\n")
            for mol, error in failed_ref_process:
                f.write(f"{mol}: {error}\n")
            f.write("\nFailed optimized molecules:\n")
            for mol, error in failed_opt_process:
                f.write(f"{mol}: {error}\n")
                
        print(f"\nProcessed data saved to {processed_dir}")
        print(f"Processing failures logged to {processed_dir}/processing_failures.txt")
        
    except Exception as e:
        print(f"Error saving processed data: {e}")
        exit(1)