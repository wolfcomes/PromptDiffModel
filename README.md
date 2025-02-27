# PromptDiff
## The program PromptDiff is a model for Structure based drug optimization. It can optimize the drug ligand in the protein pocket.

## First, I will introduce the data folder
## The data folder includes four subfolder, crossdocked_groups, docking_results, generate_data, generate_groups.
## The crossdocked_groups contains 48 groups of pocket-ligand structure from the dataset CrossDocked2020. It will be used for different prompts-opt pocket-ligand pairs generation
## the generate_data contains different prompts-opt ligand pairs.
## the generate_groups is a temp folder, contains the opt ligands' sdf file.
## the docking_results contains the opt ligand docking results with the ref pocket. And the ref-ligand, ref-pocket file.


## first, we matched the ref ligands with the origin_docking smiles
    python merge_smiles.py

## Then, we generate the opt ligands sdf files
    python sdf_generate.py

## We need to generate the docking_results first.
    python docking_multi.py

## Select the min vina score result
    python min_bind_energy_sdf.py

## We can use split_by_name_create.py file to create a .pt file for the /docking_results folder.
    python split_by_name_create.py

## The generate of the /processed_crossdock_noH_full_temp folder, contains the converted dataset and summary statistics of the docking_results file
    python process_crossdock.py ../data/docking_results/ --no_H

## We can use prompt_diff_train.py as a training example.
    python prompt_diff_train.py



