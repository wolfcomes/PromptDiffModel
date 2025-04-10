import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
import hashlib
from multiprocessing import Pool, cpu_count
import numpy as np

def generate_pair_id(source_smiles, target_smiles, idx):
    """生成分子对的唯一标识符"""
    pair_string = f"{source_smiles}{target_smiles}{idx}"
    hash_object = hashlib.md5(pair_string.encode())
    return hash_object.hexdigest()[:8]

def smi_to_molecule(smiles):
    """将SMILES转换为RDKit分子对象"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except:
        return None

def generate_3d_conformation(mol):
    """生成分子的3D构象"""
    try:
        mol_3d = Chem.AddHs(mol)
        # 使用较为快速的3D构象生成参数
        AllChem.EmbedMolecule(mol_3d, randomSeed=42, maxAttempts=1)
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=30)
        return mol_3d
    except:
        return None

def save_sdf(mol, output_path, properties=None):
    """保存分子到SDF文件，并添加属性"""
    if mol is None:
        return False
    
    try:
        writer = SDWriter(output_path)
        if properties:
            for key, value in properties.items():
                mol.SetProp(key, str(value))
        writer.write(mol)
        writer.close()
        return True
    except:
        return False

def process_molecule_pair(args):
    """处理单个分子对"""
    idx, row, source_dir, target_dir = args
    source_smiles = row['source_smiles']
    target_smiles = row['target_smiles']
    property_changes = row['property_changes']
    
    # 生成分子对的唯一标识符
    pair_id = generate_pair_id(source_smiles, target_smiles, idx)
    
    # 转换SMILES到分子对象
    source_mol = smi_to_molecule(source_smiles)
    target_mol = smi_to_molecule(target_smiles)
    
    if source_mol is None or target_mol is None:
        return None
        
    # 生成3D构象
    source_3d = generate_3d_conformation(source_mol)
    target_3d = generate_3d_conformation(target_mol)
    
    if source_3d is None or target_3d is None:
        return None
        
    # 准备属性字典
    properties = {
        'Pair_ID': pair_id,
        'SMILES': source_smiles,
        'Target_SMILES': target_smiles,
        'Property_Changes': property_changes,
        'Index': str(idx)
    }
    
    # 保存SDF文件
    source_path = os.path.join(source_dir, f'{pair_id}_source.sdf')
    target_path = os.path.join(target_dir, f'{pair_id}_target.sdf')
    
    save_success = True
    if not save_sdf(source_3d, source_path, properties):
        save_success = False
        
    if not save_sdf(target_3d, target_path, properties):
        save_success = False
    
    if save_success:
        return {
            'Pair_ID': pair_id,
            'Source_File': f'{pair_id}_source.sdf',
            'Target_File': f'{pair_id}_target.sdf',
            'Index': idx
        }
    return None

def process_admet_data(input_file, output_dir):
    """处理ADMET比较数据并生成SDF文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    source_dir = os.path.join(output_dir, 'source')
    target_dir = os.path.join(output_dir, 'target')
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    # 创建记录文件
    pair_info_file = os.path.join(output_dir, 'pair_info.csv')
    
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 准备多进程参数
    n_cores = 16  # 保留一个核心给系统
    print(f"使用 {n_cores} 个CPU核心进行并行处理")
    
    # 准备参数列表
    args_list = [(idx, row, source_dir, target_dir) 
                 for idx, row in df.iterrows()]
    
    # 使用进程池处理数据
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(process_molecule_pair, args_list),
            total=len(args_list),
            desc="处理分子对"
        ))
    
    # 过滤出成功的结果
    pair_records = [r for r in results if r is not None]
    
    # 保存配对信息到CSV文件
    pair_df = pd.DataFrame(pair_records)
    pair_df.to_csv(pair_info_file, index=False)
    print(f"配对信息已保存到：{pair_info_file}")
    print(f"成功处理 {len(pair_records)} 对分子，共 {len(df)} 对")

if __name__ == "__main__":
    input_file = "../data/optimized_ligand/admet_comparison_prompts_fg_first3.csv"
    output_dir = "../data/optimized_ligand/sdf_files_test"
    
    process_admet_data(input_file, output_dir) 