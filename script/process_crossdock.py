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


def process_ligand_and_pocket(pdbfile, sdffile,
                              atom_dict, dist_cutoff, ca_only):
    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    try:
        ligand = Chem.SDMolSupplier(str(sdffile), sanitize=False)[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')
    if ligand is None:
        print(f"Error: Failed to load ligand from {sdffile}")
    # remove H atoms if not in atom_dict, other atom types that aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    # lig_atoms = [a.GetSymbol() for a in ligand.GetAtoms()
    #              if (a.GetSymbol().capitalize() in atom_dict or a.GetSymbol() != 'H')]
    # lig_coords = np.array([list(ligand.GetConformer(0).GetAtomPosition(idx))
    #                        for idx in range(ligand.GetNumAtoms())])
    lig_atoms = []
    lig_coords=[]
    bonds_info = []  # To store bond information (atom indices and bond types)
    atom_mapping = {}
 
    for idx, a in enumerate(ligand.GetAtoms()):
    # Only include atoms that are in atom_dict or are not hydrogen
        if (a.GetSymbol().capitalize() in atom_dict or a.GetSymbol() != 'H'):
            lig_atoms.append(a.GetSymbol())  # Add atom symbol
            atom_mapping[len(lig_atoms) - 1] = idx
            # Add atom coordinates only if it's not a hydrogen atom
            if a.GetSymbol() != 'H':
                lig_coords.append(list(ligand.GetConformer(0).GetAtomPosition(idx)))

    # Convert lig_coords to a numpy array
    lig_coords = np.array(lig_coords)




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
        raise KeyError(
            f'{e} not in atom dict ({sdffile})')

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(
                    -1) ** 0.5).min() < dist_cutoff:
            pocket_residues.append(residue)

    pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in pocket_residues]
    ligand_data = {
        'lig_coords': lig_coords,
        'lig_one_hot': lig_one_hot,
        'lig_bonds': bonds_info_matrix
    }
    if ca_only:
        try:
            pocket_one_hot = []
            full_coords = []
            for res in pocket_residues:
                for atom in res.get_atoms():
                    if atom.name == 'CA':
                        pocket_one_hot.append(np.eye(1, len(amino_acid_dict),
                                                     amino_acid_dict[three_to_one(res.get_resname())]).squeeze())
                        full_coords.append(atom.coord)
            pocket_one_hot = np.stack(pocket_one_hot)
            full_coords = np.stack(full_coords)
        except KeyError as e:
            raise KeyError(
                f'{e} not in amino acid dict ({pdbfile}, {sdffile})')
        pocket_data = {
            'pocket_coords': full_coords,
            'pocket_one_hot': pocket_one_hot,
            'pocket_ids': pocket_ids
        }
    else:
        full_atoms = np.concatenate(
            [np.array([atom.element for atom in res.get_atoms()])
             for res in pocket_residues], axis=0)
        full_coords = np.concatenate(
            [np.array([atom.coord for atom in res.get_atoms()])
             for res in pocket_residues], axis=0)
        try:
            pocket_one_hot = []
            for a in full_atoms:
                if a in amino_acid_dict:
                    atom = np.eye(1, len(amino_acid_dict),
                                  amino_acid_dict[a.capitalize()]).squeeze()
                elif a != 'H':
                    atom = np.eye(1, len(amino_acid_dict),
                                  len(amino_acid_dict)).squeeze()
                pocket_one_hot.append(atom)
            pocket_one_hot = np.stack(pocket_one_hot)
        except KeyError as e:
            raise KeyError(
                f'{e} not in atom dict ({pdbfile})')
        pocket_data = {
            'pocket_coords': full_coords,
            'pocket_one_hot': pocket_one_hot,
            'pocket_ids': pocket_ids
        }
    return ligand_data, pocket_data


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


def saveall(filename, pdb_and_mol_ids, ref_lig_coords, ref_lig_one_hot,ref_lig_bonds, ref_lig_mask,
            pocket_coords, pocket_one_hot, pocket_mask, prompt_labels,opt_lig_coords,
                opt_lig_one_hot,opt_lig_bond, opt_lig_mask):
    np.savez(filename,
             names=pdb_and_mol_ids,
             prompt_labels=prompt_labels,
             ref_lig_coords=ref_lig_coords,
             ref_lig_one_hot=ref_lig_one_hot,
             ref_lig_bonds = ref_lig_bonds,
             ref_lig_mask=ref_lig_mask,
             pocket_coords=pocket_coords,
             pocket_one_hot=pocket_one_hot,
             pocket_mask=pocket_mask,
             opt_lig_coords = opt_lig_coords,
             opt_lig_one_hot = opt_lig_one_hot,
             opt_lig_bond = opt_lig_bond, 
             opt_lig_mask = opt_lig_mask
             )
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=Path)
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--no_H', action='store_true')
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--dist_cutoff', type=float, default=8.0)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    datadir = '../data/docking_results/group_2/'

    if args.ca_only:
        dataset_info = dataset_params['crossdock']
    else:
        dataset_info = dataset_params['crossdock_full']
    amino_acid_dict = dataset_info['aa_encoder']
    atom_dict = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']

    # Make output directory
    if args.outdir is None:
        suffix = '_crossdock' if 'H' in atom_dict else '_crossdock_noH'
        suffix += '_ca_only_temp' if args.ca_only else '_full_temp'
        processed_dir = Path(args.basedir, f'processed{suffix}')
    else:
        processed_dir = args.outdir

    processed_dir.mkdir(exist_ok=True, parents=True)

    # Read data split
    split_path = Path(args.basedir, 'split_by_name_2.pt')
    data_split = torch.load(split_path)

    # There is no validation set, copy 300 training examples (the validation set
    # is not very important in this application)
    # Note: before we had a data leak but it should not matter too much as most
    # metrics monitored during training are independent of the pockets
    data_split['val'] = random.sample(data_split['train'], 20)

    n_train_before = len(data_split['train'])
    n_val_before = len(data_split['val'])
    n_test_before = len(data_split['test'])

    failed_save = []

    n_samples_after = {}
    for split in data_split.keys():
        ref_lig_coords = []
        ref_lig_one_hot = []
        ref_lig_bond = []
        ref_lig_mask = []
        pocket_coords = []
        pocket_one_hot = []
        pocket_mask = []
        pdb_and_mol_ids = []
        count_protein = []
        count_ligand = []
        count_total = []
        opt_lig_coords = []
        opt_lig_one_hot = []
        opt_lig_bond = []
        opt_lig_mask = []
        count = 0

        pdb_sdf_dir = processed_dir / split
        pdb_sdf_dir.mkdir(exist_ok=True)

        tic = time()
        num_failed = 0
        pbar = tqdm(data_split[split])
        pbar.set_description(f'#failed: {num_failed}')

        for sample_pairs in pbar:
            opt_lig_id = 0
            ref_lig_id = 0
            for (pocket_fn, ligand_fn) in sample_pairs:
            
                datadir = Path(datadir)
                sdffile = datadir / f'{ligand_fn}'
                pdbfile = datadir / f'{pocket_fn}'

                if ligand_fn.endswith("minimized.sdf"):
                    ligand_type = 1
                else:
                    ligand_type = 0

                try:
                    struct_copy = PDBParser(QUIET=True).get_structure('', pdbfile)
                except:
                    num_failed += 1
                    failed_save.append((pocket_fn, ligand_fn))
                    print(failed_save[-1])
                    pbar.set_description(f'#failed: {num_failed}')
                    continue

                try:
                    ligand_data, pocket_data = process_ligand_and_pocket(
                        pdbfile, sdffile,
                        atom_dict=atom_dict, dist_cutoff=args.dist_cutoff,
                        ca_only=args.ca_only)
                except (KeyError, AssertionError, FileNotFoundError, IndexError,
                        ValueError) as e:
                    print(type(e).__name__, e, pocket_fn, ligand_fn)
                    num_failed += 1
                    pbar.set_description(f'#failed: {num_failed}')
                    continue
                
                if ligand_type == 0:
                    pdb_and_mol_ids.append(f"{pocket_fn}_{ligand_fn}")
                    ref_lig_coords.append(ligand_data['lig_coords'])
                    ref_lig_one_hot.append(ligand_data['lig_one_hot'])
                    ref_lig_bond.append(ligand_data['lig_bonds'])
                    ref_lig_mask.append(count * np.ones(len(ligand_data['lig_coords'])))
                    pocket_coords.append(pocket_data['pocket_coords'])
                    pocket_one_hot.append(pocket_data['pocket_one_hot'])
                    pocket_mask.append(
                        count * np.ones(len(pocket_data['pocket_coords'])))
                    count_protein.append(pocket_data['pocket_coords'].shape[0])
                    count_ligand.append(ligand_data['lig_coords'].shape[0])
                    count_total.append(pocket_data['pocket_coords'].shape[0] + ligand_data['lig_coords'].shape[0])
                    ref_lig_id += 1
                if ligand_type == 1:
                    
                    opt_lig_coords.append(ligand_data['lig_coords'])
                    opt_lig_one_hot.append(ligand_data['lig_one_hot'])
                    opt_lig_bond.append(ligand_data['lig_bonds'])
                    opt_lig_mask.append((count+opt_lig_id) * np.ones(len(ligand_data['lig_coords'])))
                    opt_lig_id += 1
                    
                
                if split in {'val', 'test'}:
                    # Copy PDB file
                    new_rec_name = Path(pdbfile).stem.replace('_', '-')
                    pdb_file_out = Path(pdb_sdf_dir, f"{new_rec_name}.pdb")
                    shutil.copy(pdbfile, pdb_file_out)

                    # Copy SDF file
                    new_lig_name = new_rec_name + '_' + Path(sdffile).stem.replace('_', '-')
                    sdf_file_out = Path(pdb_sdf_dir, f'{new_lig_name}.sdf')
                    shutil.copy(sdffile, sdf_file_out)

                    # specify pocket residues
                    with open(Path(pdb_sdf_dir, f'{new_lig_name}.txt'), 'w') as f:
                        f.write(' '.join(pocket_data['pocket_ids']))
            
            
            # sample_pairs 循环结束后检查 opt 部分是否为空
            if opt_lig_id == 0 and ref_lig_id != 0:
                for i in range(ref_lig_id):
                    ref_lig_coords.pop()
                    ref_lig_one_hot.pop()
                    ref_lig_bond.pop()
                    ref_lig_mask.pop()
                    pocket_coords.pop()
                    pocket_one_hot.pop()
                    pocket_mask.pop()
                    pdb_and_mol_ids.pop()
                    count_protein.pop()
                    count_ligand.pop()
                    count_total.pop()
            

                
            
            if opt_lig_id >= 2:
                # 复制最后一个元素 `opt_lig_id - 1` 次
                for j in range(1,opt_lig_id):
                    ref_lig_coords.append(ref_lig_coords[-1])
                    ref_lig_one_hot.append(ref_lig_one_hot[-1])
                    ref_lig_bond.append(ref_lig_bond[-1])
                    ref_lig_mask.append(ref_lig_mask[-1]+1)
                    pocket_coords.append(pocket_coords[-1])
                    pocket_one_hot.append(pocket_one_hot[-1])
                    pocket_mask.append(pocket_mask[-1]+1)
                    pdb_and_mol_ids.append(pdb_and_mol_ids[-1])
                    count_protein.append(count_protein[-1])
                    count_ligand.append(count_ligand[-1])
                    count_total.append(count_total[-1])
            
            if opt_lig_id != 0 and ref_lig_id == 0:
                for i in range(opt_lig_id):
                    opt_lig_coords.pop()
                    opt_lig_one_hot.pop()
                    opt_lig_bond.pop()
                    opt_lig_mask.pop()
                count -= opt_lig_id

            count += opt_lig_id

            # for i in range(len(opt_lig_coords)):
            #     if len(opt_lig_coords[i]) > 0:
            #         print(len(opt_lig_coords[i]))
            #         opt_lig_coords[i] = np.concatenate(opt_lig_coords[i], axis=0)
            #         opt_lig_one_hot[i] = np.concatenate(opt_lig_one_hot[i], axis=0)
            #         opt_lig_bond[i] = np.concatenate(opt_lig_bond[i], axis=0)
            #         #opt_lig_mask[i] = np.concatenate(opt_lig_mask[i], axis=0)
            #     else:
            #         # 假设每个 ligand 坐标的形状是 (n, 3)
            #         opt_lig_coords[i] = np.empty((0, 3))
            #         opt_lig_one_hot[i] = np.empty((0, 10))
            #         opt_lig_bond[i] = np.empty((0, 2))
            #         #opt_lig_mask[i] = np.empty((0,))




        print(len(ref_lig_coords))
        print(len(ref_lig_one_hot))
        ref_lig_coords = np.concatenate(ref_lig_coords, axis=0)
        ref_lig_one_hot = np.concatenate(ref_lig_one_hot, axis=0)
        ref_lig_bond = np.concatenate(ref_lig_bond, axis=0)
        ref_lig_mask = np.concatenate(ref_lig_mask, axis=0)
        
        pocket_coords = np.concatenate(pocket_coords, axis=0)
        pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
        pocket_mask = np.concatenate(pocket_mask, axis=0)
        opt_lig_coords = np.concatenate(opt_lig_coords, axis=0)
        opt_lig_one_hot = np.concatenate(opt_lig_one_hot, axis=0)
        opt_lig_bond = np.concatenate(opt_lig_bond, axis=0)
        opt_lig_mask = np.concatenate(opt_lig_mask, axis=0)
        prompt_labels = np.tile([0, 0, 1], (len(opt_lig_coords), 1))
        print(len(ref_lig_coords))
        print(len(ref_lig_one_hot))


        saveall(processed_dir / f'{split}.npz', pdb_and_mol_ids, ref_lig_coords,
                ref_lig_one_hot,ref_lig_bond, ref_lig_mask, pocket_coords,
                pocket_one_hot, pocket_mask, prompt_labels, opt_lig_coords,
                opt_lig_one_hot,opt_lig_bond, opt_lig_mask)

        n_samples_after[split] = len(pdb_and_mol_ids)
        print(f"Processing {split} set took {(time() - tic) / 60.0:.2f} minutes")

    # --------------------------------------------------------------------------
    # Compute statistics & additional information
    # --------------------------------------------------------------------------
    with np.load(processed_dir / 'train.npz', allow_pickle=True) as data:
        ref_lig_mask = data['ref_lig_mask']
        ref_lig_coords = data['ref_lig_coords']
        ref_lig_one_hot = data['ref_lig_one_hot']
        ref_lig_bonds = data['ref_lig_bonds']
        opt_lig_mask = data['opt_lig_mask']
        opt_lig_coords = data['opt_lig_coords']
        opt_lig_one_hot = data['opt_lig_one_hot']
        opt_lig_bond = data['opt_lig_bond']
        pocket_mask = data['pocket_mask']
        pocket_one_hot = data['pocket_one_hot']

    # Compute SMILES for all training examples
    train_smiles = compute_smiles(opt_lig_coords, opt_lig_one_hot, opt_lig_mask)
    np.save(processed_dir / 'train_smiles.npy', train_smiles)

    # Joint histogram of number of ligand and pocket nodes
    n_nodes = get_n_nodes(opt_lig_mask, pocket_mask, smooth_sigma=1.0)
    np.save(Path(processed_dir, 'size_distribution.npy'), n_nodes)

    # Convert bond length dictionaries to arrays for batch processing
    bonds1, bonds2, bonds3 = get_bond_length_arrays(atom_dict)

    # Get bond length definitions for Lennard-Jones potential
    rm_LJ = get_lennard_jones_rm(atom_dict)

    # Get histograms of ligand and pocket node types
    atom_hist, aa_hist = get_type_histograms(opt_lig_one_hot, pocket_one_hot,
                                             atom_dict, amino_acid_dict)

    # Create summary string
    summary_string = '# SUMMARY\n\n'
    summary_string += '# Before processing\n'
    summary_string += f'num_samples train: {n_train_before}\n'
    summary_string += f'num_samples val: {n_val_before}\n'
    summary_string += f'num_samples test: {n_test_before}\n\n'
    summary_string += '# After processing\n'
    summary_string += f"num_samples train: {n_samples_after['train']}\n"
    summary_string += f"num_samples val: {n_samples_after['val']}\n"
    summary_string += f"num_samples test: {n_samples_after['test']}\n\n"
    summary_string += '# Info\n'
    summary_string += f"'atom_encoder': {atom_dict}\n"
    summary_string += f"'atom_decoder': {list(atom_dict.keys())}\n"
    summary_string += f"'aa_encoder': {amino_acid_dict}\n"
    summary_string += f"'aa_decoder': {list(amino_acid_dict.keys())}\n"
    summary_string += f"'bonds1': {bonds1.tolist()}\n"
    summary_string += f"'bonds2': {bonds2.tolist()}\n"
    summary_string += f"'bonds3': {bonds3.tolist()}\n"
    summary_string += f"'lennard_jones_rm': {rm_LJ.tolist()}\n"
    summary_string += f"'atom_hist': {atom_hist}\n"
    summary_string += f"'aa_hist': {aa_hist}\n"
    summary_string += f"'n_nodes': {n_nodes.tolist()}\n"

    sns.distplot(count_protein)
    plt.savefig(processed_dir / 'protein_size_distribution.png')
    plt.clf()

    sns.distplot(count_ligand)
    plt.savefig(processed_dir / 'lig_size_distribution.png')
    plt.clf()

    sns.distplot(count_total)
    plt.savefig(processed_dir / 'total_size_distribution.png')
    plt.clf()

    # Write summary to text file
    with open(processed_dir / 'summary.txt', 'w') as f:
        f.write(summary_string)

    # Print summary
    print(summary_string)

    print(failed_save)
