import os
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
import argparse
import random
from collections import defaultdict

def process_ligand(sdf_file, atom_dict):
    """处理单个配体分子"""
    try:
        ligand = Chem.SDMolSupplier(str(sdf_file), sanitize=False)[0]
        if ligand is None:
            return None
            
        # 提取属性
        properties = {
            'Pair_ID': ligand.GetProp('Pair_ID'),
            'SMILES': ligand.GetProp('SMILES'),
            'Index': int(ligand.GetProp('Index'))
        }
        
        lig_atoms = []
        lig_coords = []
        atom_mapping = {}
        
        for idx, a in enumerate(ligand.GetAtoms()):
            if a.GetSymbol().capitalize() in atom_dict:
                lig_atoms.append(a.GetSymbol())
                atom_mapping[len(lig_atoms) - 1] = idx
                lig_coords.append(list(ligand.GetConformer(0).GetAtomPosition(idx)))

        lig_coords = np.array(lig_coords)

        # 处理非氢原子的键信息
        non_h_atoms = []
        for idx, a in enumerate(ligand.GetAtoms()):
            atom_symbol = a.GetSymbol().capitalize()
            if atom_symbol != 'H':
                non_h_atoms.append({
                    'idx': idx,
                    'symbol': atom_symbol
                })

        bond_type_map = {
            'NONE': [1, 0, 0, 0, 0, 0, 0],
            'SINGLE': [0, 1, 0, 0, 0, 0, 0],
            'DOUBLE': [0, 0, 1, 0, 0, 0, 0],
            'TRIPLE': [0, 0, 0, 1, 0, 0, 0],
            'AROMATIC': [0, 0, 0, 0, 1, 0, 0],
            'ANY': [0, 0, 0, 0, 0, 1, 0],
            'SELF': [0, 0, 0, 0, 0, 0, 1]
        }

        N = len(non_h_atoms)
        max_lig_num = 100
        bonds_info_matrix = np.zeros((N, max_lig_num, 7), dtype=int)

        # 构建键信息矩阵
        for i in range(N):
            for j in range(i + 1, N):
                atom1_idx = non_h_atoms[i]['idx']
                atom2_idx = non_h_atoms[j]['idx']
                
                bond_found = False
                for bond in ligand.GetBonds():
                    if (bond.GetBeginAtomIdx() == atom1_idx and bond.GetEndAtomIdx() == atom2_idx) or \
                    (bond.GetBeginAtomIdx() == atom2_idx and bond.GetEndAtomIdx() == atom1_idx):
                        bond_type_num = bond_type_map.get(bond.GetBondType().name, bond_type_map['ANY'])
                        bonds_info_matrix[i, j] = bond_type_num
                        bonds_info_matrix[j, i] = bond_type_num
                        bond_found = True
                        break
                
                if not bond_found:
                    bonds_info_matrix[i, j] = bond_type_map['NONE']
                    bonds_info_matrix[j, i] = bond_type_map['NONE']

        # 处理自连接
        for k in range(len(non_h_atoms)):
            bonds_info_matrix[k, k] = bond_type_map['SELF']

        # 填充剩余的空位
        for i in range(N):
            for j in range(i + 1, max_lig_num):
                if np.array_equal(bonds_info_matrix[i, j], np.array([0, 0, 0, 0, 0, 0, 0])):
                    bonds_info_matrix[i, j] = bond_type_map['NONE']

        # 生成one-hot编码
        try:
            lig_one_hot = np.stack([
                np.eye(1, len(atom_dict), atom_dict[a.capitalize()]).squeeze()
                for a in lig_atoms
            ])
        except KeyError as e:
            print(f'Atom type error in {sdf_file}: {e}')
            return None

        return {
            'coords': lig_coords,
            'one_hot': lig_one_hot,
            'bonds': bonds_info_matrix,
            'properties': properties
        }
        
    except Exception as e:
        print(f'Error processing {sdf_file}: {e}')
        return None

def save_processed_data(filename, data_dict):
    """保存处理后的数据"""
    np.savez(filename,
             names=data_dict['mol_ids'],
             prompt_labels=data_dict['prompt_labels'],
             ref_lig_coords=data_dict['ref_lig_coords'],
             ref_lig_one_hot=data_dict['ref_lig_one_hot'],
             ref_lig_bonds=data_dict['ref_lig_bonds'],
             ref_lig_mask=data_dict['ref_lig_mask'],
             opt_lig_coords=data_dict['opt_lig_coords'],
             opt_lig_one_hot=data_dict['opt_lig_one_hot'],
             opt_lig_bond=data_dict['opt_lig_bonds'],
             opt_lig_mask=data_dict['opt_lig_mask'],
             pocket_coords=data_dict['pocket_coords'],
             pocket_one_hot=data_dict['pocket_one_hot'],
             pocket_mask=data_dict['pocket_mask']
             )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf_dir', type=Path, required=True, help='Directory containing source and target SDF files')
    parser.add_argument('--outdir', type=Path, required=True, help='Output directory for processed files')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.1)
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # 创建输出目录
    args.outdir.mkdir(exist_ok=True, parents=True)

    # 加载数据集参数
    from constants import dataset_params
    dataset_info = dataset_params['crossdock_full']
    atom_dict = dataset_info['atom_encoder']
    aa_encoder = dataset_info['aa_encoder']

    # 读取pair_info.csv
    pair_info = pd.read_csv(args.sdf_dir / 'pair_info.csv')
    
    # 用于存储处理后的数据
    processed_pairs = []
    failed_pairs = []
    
    print("Processing molecule pairs...")
    for _, row in tqdm(pair_info.iterrows(), total=len(pair_info)):
        source_file = args.sdf_dir / 'source' / row['Source_File']
        target_file = args.sdf_dir / 'target' / row['Target_File']
        
        # 处理source和target分子
        source_data = process_ligand(source_file, atom_dict)
        target_data = process_ligand(target_file, atom_dict)
        
        if source_data is None or target_data is None:
            failed_pairs.append(row['Pair_ID'])
            continue
            
        processed_pairs.append({
            'pair_id': row['Pair_ID'],
            'source': source_data,
            'target': target_data,
            'index': row['Index']
        })

    print(f"Successfully processed {len(processed_pairs)} pairs")
    print(f"Failed to process {len(failed_pairs)} pairs")

    # 准备数据集划分
    indices = list(range(len(processed_pairs)))
    random.shuffle(indices)

    test_size = int(len(indices) * args.test_size)
    val_size = int(len(indices) * args.val_size)
    train_size = len(indices) - test_size - val_size

    splits = {
        'train': indices[:train_size],
        'val': indices[train_size:train_size + val_size],
        'test': indices[train_size + val_size:]
    }

    # 处理每个数据集划分
    for split_name, split_indices in splits.items():
        if len(split_indices) == 0:
            continue

        split_pairs = [processed_pairs[i] for i in split_indices]
        
        # 准备数据字典
        data_dict = {
            'mol_ids': [],
            'ref_lig_coords': [],
            'ref_lig_one_hot': [],
            'ref_lig_bonds': [],
            'ref_lig_mask': [],
            'opt_lig_coords': [],
            'opt_lig_one_hot': [],
            'opt_lig_bonds': [],
            'opt_lig_mask': [],
            'pocket_coords': [],
            'pocket_one_hot': [],
            'pocket_mask': []
        }

        for idx, pair in enumerate(split_pairs):
            data_dict['mol_ids'].append(f"{pair['pair_id']}")
            data_dict['ref_lig_coords'].append(pair['source']['coords'])
            data_dict['ref_lig_one_hot'].append(pair['source']['one_hot'])
            data_dict['ref_lig_bonds'].append(pair['source']['bonds'])
            data_dict['ref_lig_mask'].append(idx * np.ones(len(pair['source']['coords'])))
            data_dict['opt_lig_coords'].append(pair['target']['coords'])
            data_dict['opt_lig_one_hot'].append(pair['target']['one_hot'])
            data_dict['opt_lig_bonds'].append(pair['target']['bonds'])
            data_dict['opt_lig_mask'].append(idx * np.ones(len(pair['target']['coords'])))
            data_dict['pocket_coords'].append(np.array([[0.0, 0.0, 0.0]]))
            data_dict['pocket_one_hot'].append(np.eye(1, len(aa_encoder), len(aa_encoder)-1))
            data_dict['pocket_mask'].append(idx * np.ones(1))

        # 转换为numpy数组
        for key in ['ref_lig_coords', 'ref_lig_one_hot', 'ref_lig_bonds', 'ref_lig_mask',
                   'opt_lig_coords', 'opt_lig_one_hot', 'opt_lig_bonds', 'opt_lig_mask',
                   'pocket_coords', 'pocket_one_hot', 'pocket_mask']:
            data_dict[key] = np.concatenate(data_dict[key], axis=0)

        # 添加prompt_labels
        data_dict['prompt_labels'] = np.tile([0, 0, 1], (len(data_dict['opt_lig_coords']), 1))

        # 保存数据
        save_processed_data(args.outdir / f'{split_name}.npz', data_dict)
        print(f"Saved {split_name} set with {len(split_pairs)} pairs")

    # 保存处理失败的记录
    with open(args.outdir / 'processing_failures.txt', 'w') as f:
        f.write("Failed pairs:\n")
        for pair_id in failed_pairs:
            f.write(f"{pair_id}\n")

if __name__ == '__main__':
    main() 