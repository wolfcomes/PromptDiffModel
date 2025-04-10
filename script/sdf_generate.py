import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolAlign
from tqdm import tqdm  # Import tqdm for progress bar
from multiprocessing import Pool, cpu_count  # 添加多核处理支持

# Function to read SDF and SMILES, create molecule object from SMILES
def smi_to_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except:
        return None

# Function to calculate RMSD between two molecules
def calculate_rmsd(mol1, mol2):
    try:
        # Generate 3D coordinates for both molecules if they don't have them
        mol1_3d = Chem.AddHs(mol1)
        mol2_3d = Chem.AddHs(mol2)
        
        AllChem.EmbedMolecule(mol1_3d, randomSeed=42)
        AllChem.EmbedMolecule(mol2_3d, randomSeed=42)
        
        # Perform alignment to minimize RMSD
        rmsd = rdMolAlign.GetBestRMS(mol1_3d, mol2_3d)
        return rmsd
    except:
        return float('inf')  # If the RMSD calculation fails, return a large value

# Function to align optimized molecule to reference molecule
def align_molecules(mol1, mol2):
    try:
        # Generate 3D coordinates if not present
        mol1_3d = Chem.AddHs(mol1)
        mol2_3d = Chem.AddHs(mol2)

        # Embed the molecules to generate 3D coordinates if not already available
        AllChem.EmbedMolecule(mol1_3d, randomSeed=42)
        AllChem.EmbedMolecule(mol2_3d, randomSeed=42)

        # Perform alignment based on 3D structures
        rmsd = rdMolAlign.GetBestRMS(mol2_3d, mol1_3d)
    except:
        return mol2_3d, float('inf')
    return mol2_3d, rmsd

# Function to generate SDF file from aligned molecule
def generate_sdf_from_molecule(output_file, molecule):
    writer = SDWriter(output_file)
    writer.write(molecule)
    writer.close()

# Function to find the best matching SDF file based on SDF File name
def find_matching_sdf_file(reference_dir, sdf_file_name):
    # Match the reference file by checking if the file starts with sdf_file_name and ends with .sdf
    for sdf_filename in os.listdir(reference_dir):
        if sdf_filename.startswith(sdf_file_name) and sdf_filename.endswith(".sdf"):
            return os.path.join(reference_dir, sdf_filename)
    return None

# 添加一个处理单个分子的函数，便于并行
def process_single_molecule(args):
    idx, row, reference_dir, output_dir = args
    # 在try块外部先初始化变量，防止异常处理中引用未定义变量
    sdf_file_name = "unknown"
    try:
        standardized_opt_smi = row['target_smiles']  # 使用target_smiles作为目标SMILES
        sdf_file_name = row['SDF File']
        file_prefix = '_'.join(sdf_file_name.split('_')[:8])
        
        # 获取属性变化信息
        property_changes = row.get('property_changes', '')

        reference_sdf_dir = reference_dir + '/' + file_prefix
        # Find the best matching SDF file based on SDF File column in CSV
        matching_sdf_path = find_matching_sdf_file(reference_sdf_dir, sdf_file_name)
        
        if not matching_sdf_path:
            return f"No matching SDF file found for {sdf_file_name}. Skipping."
            
        # Read the reference molecule from the matching SDF file
        suppl = Chem.SDMolSupplier(matching_sdf_path)
        reference_mol = None
        for mol in suppl:
            if mol is not None:
                reference_mol = mol
                break
        
        if reference_mol is None:
            return f"No valid molecule found in the file {matching_sdf_path}. Skipping."
        
        # Generate the target molecule from the standardized opt_smi
        target_mol = smi_to_molecule(standardized_opt_smi)
        if target_mol is None:
            return f"Invalid SMILES for {sdf_file_name}. Skipping."
        
        # Align the molecules
        aligned_mol, rmsd = align_molecules(reference_mol, target_mol)
        
        # 添加属性变化信息到分子对象
        if property_changes and not pd.isna(property_changes):
            aligned_mol.SetProp("PropertyChanges", property_changes)
        
        # Create output directory if it doesn't exist
        output_subdir = os.path.join(output_dir, file_prefix)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Get unique file name
        existing_files = [f for f in os.listdir(output_subdir) if f.startswith(file_prefix)]
        file_index = len(existing_files) + 1
        output_path = os.path.join(output_subdir, f"{file_prefix}_generated_{file_index}.sdf")
        
        # Generate output SDF file
        generate_sdf_from_molecule(output_path, aligned_mol)
        return f"Generated SDF file: {output_path} with RMSD: {rmsd:.4f} | PropertyChanges: {'Yes' if property_changes and not pd.isna(property_changes) else 'No'}"
    except Exception as e:
        return f"Error processing {sdf_file_name} (row {idx}): {str(e)}"

def process_csv_and_generate_sdfs(csv_file, reference_dir, output_dir, n_cores=None):
    df = pd.read_csv(csv_file)
    print(f"读取了 {len(df)} 条记录")
    
    # 确定使用的核心数
    if n_cores is None:
        n_cores = max(1, cpu_count() - 1)  # 默认使用CPU核心数-1
    print(f"使用 {n_cores} 个CPU核心进行并行处理")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备参数列表
    args_list = [(idx, row, reference_dir, output_dir) for idx, row in df.iterrows()]
    
    # 使用多核并行处理
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(process_single_molecule, args_list),
            total=len(args_list),
            desc="Processing molecules"
        ))
    
    # 打印结果
    for result in results:
        print(result)
    
    success_count = sum(1 for r in results if r.startswith("Generated"))
    print(f"成功生成: {success_count}/{len(df)} 个SDF文件")

# Example usage
if __name__ == "__main__":

    for n in range(3, 4):
        # Define your paths
        csv_file = "../data/optimized_ligand/merged_all_groups_admet.csv"  # Replace with your actual CSV file path
        reference_dir = "../data/crossdocked_groups/group_" + str(n)  # Replace with your reference SDF directory
        output_dir = "../data/generate_groups/group_" + str(n)  # Replace with your desired output directory
        
        # 设置并行核心数，根据实际情况调整
        n_cores = 48  # 可以根据系统情况调整
    
        process_csv_and_generate_sdfs(csv_file, reference_dir, output_dir, n_cores)