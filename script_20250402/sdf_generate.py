import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolAlign
from tqdm import tqdm  # Import tqdm for progress bar

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

def process_csv_and_generate_sdfs(csv_file, reference_dir, output_dir):
    df = pd.read_csv(csv_file)
    
    # Use tqdm to create a progress bar for iterating over rows in the DataFrame
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
        standardized_opt_smi = row['standardized_opt_smi']
        sdf_file_name = row['SDF File']
        file_prefix = '_'.join(sdf_file_name.split('_')[:8])

        reference_sdf_dir = reference_dir + '/' + file_prefix
        # Find the best matching SDF file based on SDF File column in CSV
        matching_sdf_path = find_matching_sdf_file(reference_sdf_dir, sdf_file_name)
        
        if matching_sdf_path:
            # Read the reference molecule from the matching SDF file
            suppl = Chem.SDMolSupplier(matching_sdf_path)
            reference_mol = None
            for mol in suppl:
                if mol is not None:
                    reference_mol = mol
                    break
            
            if reference_mol is None:
                print(f"No valid molecule found in the file {matching_sdf_path}. Skipping.")
                continue
            
            # Generate the target molecule from the standardized opt_smi
            target_mol = smi_to_molecule(standardized_opt_smi)
            if target_mol is None:
                print(f"Invalid SMILES for {sdf_file_name}. Skipping.")
                continue
            
            # Align the molecules
            aligned_mol, rmsd = align_molecules(reference_mol, target_mol)
            
            # Unique file naming if multiple optimized structures exist for the same reference SDF
            
            if not os.path.exists(os.path.join(output_dir, file_prefix)):
                os.makedirs(os.path.join(output_dir, file_prefix))
            existing_files = [f for f in os.listdir(os.path.join(output_dir, file_prefix)) if f.startswith(file_prefix)]
            file_index = len(existing_files) + 1
            output_path = os.path.join(output_dir, file_prefix, f"{file_prefix}_generated_{file_index}.sdf")
            
            # Generate output SDF file
            generate_sdf_from_molecule(output_path, aligned_mol)
            print(f"Generated SDF file: {output_path} with RMSD: {rmsd}")
        else:
            print(f"No matching SDF file found for {sdf_file_name}. Skipping.")

# Example usage
if __name__ == "__main__":
    # Define your paths
    csv_file = "../data/generate_data/TPSA_data_all.csv"  # Replace with your actual CSV file path
    reference_dir = "../data/crossdocked_groups/group_2"  # Replace with your reference SDF directory
    output_dir = "../data/generate_groups/group_2"  # Replace with your desired output directory
    
    process_csv_and_generate_sdfs(csv_file, reference_dir, output_dir)