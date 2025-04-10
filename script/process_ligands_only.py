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
        # 读取SDF文件中的分子
        suppl = Chem.SDMolSupplier(str(sdffile), sanitize=False)
        ligand = suppl[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')
    if ligand is None:
        print(f"Error: Failed to load ligand from {sdffile}")
        
    # 从SDF文件提取PropertyChanges数据
    property_changes = None
    if ligand.HasProp("PropertyChanges"):
        property_changes = ligand.GetProp("PropertyChanges")
    
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

    # 提取SMILES字符串
    smiles = Chem.MolToSmiles(ligand)

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
        'lig_bonds': bonds_info_matrix,
        'smiles': smiles,  # SMILES字符串
        'property_changes': property_changes  # 添加属性变化数据
    }
    
    return ligand_data

def generate_prompt_labels(property_changes, all_properties=None):
    """
    根据property_changes字符串生成对应的prompt_labels
    
    Args:
        property_changes: 属性变化字符串，如"AMES:1 BBB_Martins:-1 ..."
        all_properties: 要包含的所有属性列表，如果为None，则使用默认值
        
    Returns:
        numpy数组形式的prompt_labels
    """
    # 解析property_changes
    properties = {}
    
    if property_changes:
        for item in property_changes.split():
            if ':' in item:
                prop, value = item.split(':')
                properties[prop] = int(value)
    
    # 如果未指定关注的属性列表，则使用默认值
    if all_properties is None:
        all_properties = ["Solubility_AqSolDB", "Lipophilicity_AstraZeneca", "Caco2_Wang", "PAMPA_NCATS", 
                          "HIA_Hou", "Pgp_Broccatelli", "BBB_Martins", "Bioavailability_Ma", "HydrationFreeEnergy_FreeSolv"]
    
    # 初始化空的prompt_labels模板（长度为属性数量）
    prompt_labels = np.zeros(len(all_properties) + 1)  # 加1是因为最后一位表示"提高总体属性"
    
    # 统计变化（提高、降低、不变）
    improved = 0
    decreased = 0
    unchanged = 0
    
    # 填充各属性的变化
    for i, prop in enumerate(all_properties):
        if prop in properties:
            value = properties[prop]
            prompt_labels[i] = value
            if value > 0:
                improved += 1
            elif value < 0:
                decreased += 1
            else:
                unchanged += 1
    
    # 最后一位表示总体情况：如果改进的属性多于降低的，设为1，否则设为0
    prompt_labels[-1] = 1 if improved > decreased else 0
    
    return prompt_labels

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
    parser.add_argument('--ref_dir', type=Path, nargs='+', help='Directory or multiple directories containing reference molecules')
    parser.add_argument('--opt_dir', type=Path, nargs='+', help='Directory or multiple directories containing optimized molecules')
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size ratio')
    parser.add_argument('--test_size', type=float, default=0.1, help='Test set size ratio')
    args = parser.parse_args()

    # 检查ref_dir和opt_dir参数数量是否一致
    if len(args.ref_dir) != len(args.opt_dir):
        raise ValueError("参考分子目录和优化分子目录的数量必须相同")

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
        'mol_ids': [],
        'ref_smiles': [],  # 存储参考分子的SMILES
        'opt_smiles': [],  # 存储优化分子的SMILES
        'property_changes': []  # 存储属性变化数据
    }
    
    count = 0
    
    # 记录失败的情况
    failed_pairs = []
    failed_ref_process = []
    failed_opt_process = []
    skipped_too_many_ref = []
    skipped_no_opt = []

    # 记录找到property_changes的情况
    with_property_changes = 0
    without_property_changes = 0

    # 收集所有唯一的属性名称
    all_property_names = set()

    # 处理所有组
    for group_idx, (ref_dir, opt_dir) in enumerate(zip(args.ref_dir, args.opt_dir)):
        print(f"\n处理组 {group_idx+1}/{len(args.ref_dir)}: {ref_dir.name}")
        
        # 获取所有子目录
        ref_subdirs = [d for d in ref_dir.iterdir() if d.is_dir()]
        print(f"在组 {ref_dir.name} 中找到 {len(ref_subdirs)} 个子目录")
        
        # 处理所有分子数据
        for ref_subdir in tqdm(ref_subdirs, desc=f"处理组 {ref_dir.name} 中的子目录"):
            # 检查ref_subdir中的SDF文件数量，如果有两个及以上，跳过
            ref_files = list(ref_subdir.glob('*.sdf'))
            if len(ref_files) >= 2:
                skipped_too_many_ref.append(f"{ref_dir.name}/{ref_subdir.name}")
                print(f"跳过 {ref_subdir.name}: 包含 {len(ref_files)} 个SDF文件 (>= 2)")
                continue
                
            opt_subdir = opt_dir / ref_subdir.name
            if not opt_subdir.exists():
                print(f"警告: 没有找到匹配的优化目录 {ref_subdir}")
                continue
                
            # 检查opt_subdir中是否有SDF文件，如果没有，跳过
            opt_files = list(opt_subdir.glob('*.sdf'))
            if not opt_files:
                skipped_no_opt.append(f"{ref_dir.name}/{ref_subdir.name}")
                print(f"跳过 {ref_subdir.name}: 没有找到优化分子")
                continue

            print(f"\n处理子目录 {ref_dir.name}/{ref_subdir.name}: {len(ref_files)} 个参考分子")
            
            for ref_file in ref_files:
                try:
                    try:
                        ref_data = process_ligand(ref_file, atom_dict)
                    except Exception as e:
                        failed_ref_process.append((f"{ref_dir.name}/{ref_file}", str(e)))
                        print(f"处理参考分子时出错 {ref_file}: {e}")
                        continue

                    ref_smiles = ref_data.get('smiles', '')
                    
                    for opt_file in opt_files:
                        try:
                            opt_data = process_ligand(opt_file, atom_dict)
                        except Exception as e:
                            failed_opt_process.append((f"{opt_dir.name}/{opt_file}", str(e)))
                            print(f"处理优化分子时出错 {opt_file}: {e}")
                            continue

                        opt_smiles = opt_data.get('smiles', '')
                        
                        # 从优化配体中获取属性变化数据
                        opt_property_changes = opt_data.get('property_changes', None)
                        
                        # 如果找到属性变化数据，解析其中包含的属性名称
                        if opt_property_changes:
                            with_property_changes += 1
                            for item in opt_property_changes.split():
                                if ':' in item:
                                    prop, _ = item.split(':')
                                    all_property_names.add(prop)
                        else:
                            without_property_changes += 1
                            print(f"警告: 未找到属性变化数据: {opt_file}")

                        # 添加组信息到分子ID中
                        mol_id = f"{ref_dir.name}/{ref_subdir.name}/{ref_file.stem}->{opt_file.stem}"
                        all_data['mol_ids'].append(mol_id)
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
                        all_data['ref_smiles'].append(ref_smiles)
                        all_data['opt_smiles'].append(opt_smiles)
                        all_data['property_changes'].append(opt_property_changes)
                        count += 1
                    
                except Exception as e:
                    print(f"处理 {ref_file} 时发生意外错误: {e}")
                    continue

    print(f"\n处理摘要:")
    print(f"处理的总组数: {len(args.ref_dir)}")
    print(f"成功处理的分子对: {count}")
    print(f"未找到优化对: {len(failed_pairs)}")
    print(f"处理参考分子失败: {len(failed_ref_process)}")
    print(f"处理优化分子失败: {len(failed_opt_process)}")
    print(f"跳过含有过多参考分子的目录: {len(skipped_too_many_ref)}")
    print(f"跳过没有优化分子的目录: {len(skipped_no_opt)}")
    print(f"带有属性变化数据的分子: {with_property_changes}")
    print(f"没有属性变化数据的分子: {without_property_changes}")
    print(f"发现的属性种类: {len(all_property_names)}")
    if all_property_names:
        print(f"属性列表: {', '.join(sorted(all_property_names))}")

    if count == 0:
        print("没有有效的分子对被处理。退出...")
        exit(1)

    # 将收集到的属性名称转换为排序列表
    all_properties = sorted(list(all_property_names))

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
        
        print(f"\n数据集划分:")
        print(f"训练集: {len(splits['train'])} 个样本")
        print(f"验证集: {len(splits['val'])} 个样本")
        print(f"测试集: {len(splits['test'])} 个样本")
        
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
                'ref_smiles': [all_data['ref_smiles'][i] for i in split_indices],
                'opt_smiles': [all_data['opt_smiles'][i] for i in split_indices],
                'property_changes': [all_data['property_changes'][i] for i in split_indices],
            }
            
            # 生成prompt_labels
            prompt_labels = []
            for property_changes in split_data['property_changes']:
                # 生成标签
                label = generate_prompt_labels(property_changes, all_properties)
                prompt_labels.append(label)
            
            prompt_labels = np.array(prompt_labels)
            
            
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
            f.write("处理的组目录:\n")
            for i, (ref_dir, opt_dir) in enumerate(zip(args.ref_dir, args.opt_dir)):
                f.write(f"组 {i+1}: {ref_dir} -> {opt_dir}\n")
            
            f.write("\n未找到优化对:\n")
            for pair in failed_pairs:
                f.write(f"{pair}\n")
            f.write("\n处理参考分子失败:\n")
            for mol, error in failed_ref_process:
                f.write(f"{mol}: {error}\n")
            f.write("\n处理优化分子失败:\n")
            for mol, error in failed_opt_process:
                f.write(f"{mol}: {error}\n")
            f.write("\n跳过含有过多参考分子的目录:\n")
            for dir_path in skipped_too_many_ref:
                f.write(f"{dir_path}\n")
            f.write("\n跳过没有优化分子的目录:\n")
            for dir_path in skipped_no_opt:
                f.write(f"{dir_path}\n")
            f.write("\n属性变化数据统计:\n")
            f.write(f"带有属性变化数据的分子: {with_property_changes}\n")
            f.write(f"没有属性变化数据的分子: {without_property_changes}\n")
            if all_properties:
                f.write(f"发现的属性种类: {len(all_properties)}\n")
                f.write(f"属性列表: {', '.join(all_properties)}\n")
                
        print(f"\n处理数据已保存到 {processed_dir}")
        print(f"处理失败记录已保存到 {processed_dir}/processing_failures.txt")
        
    except Exception as e:
        print(f"保存处理数据时出错: {e}")
        exit(1)